import numpy as np
from predictors import Predictor
import json
from utils import load_list_to_csv,read_list_from_csv,strided_app
import torch
import os
import numpy as np
from tqdm import *


class Encoder:
    def __init__(self,mypredictor:Predictor):
        self.predictor = mypredictor
        

    def encode(self, data_series, timesteps, slide, compressed_file_name:str, batch_size):
        
        self.compressed_file_name = compressed_file_name 
        self.timesteps = timesteps
        self.slide = slide
        self.data = data_series
        

        # deal with datasets
        len_series = len(data_series)
        if (len_series - timesteps) % slide == 0:
            ind = (len_series - timesteps) // slide
        else:
            ind = (len_series - timesteps) // slide + 1
        rem = ind * slide - (len_series - timesteps)
        if self.predictor.args.data_flag == 1:
            data_series = np.concatenate((data_series,np.full(rem,0,dtype=int)))
        else:
            data_series = np.concatenate((data_series,np.full(rem,0,dtype=float)))
        time_series = strided_app(data_series,timesteps,slide)
        X = time_series[:,:-slide]
        Y = time_series[:,-slide:]


        # params
        params_name = compressed_file_name+"params"
        params = {}
        params['len_series'] = len_series
        params['bs'] = batch_size
        params['timesteps'] = timesteps
        params['slide'] = slide
        
        with open(params_name,'w') as f:
            json.dump(params, f, indent=4)


        # create temp dir to store first timesteps of every batch
        temp_dir = self.compressed_file_name + 'temp'
        if os.path.exists(temp_dir):
            os.system("rm -r {}".format(temp_dir))
        self.temp_file_prefix = temp_dir + "/compressed"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # compress
        print("Compressing...")
        if self.predictor.args.data_flag == 1:
            residuals_list = np.zeros_like(Y,dtype=int)
        else:
            residuals_list = np.zeros_like(Y,dtype=float)
        if len(X) % batch_size > 0:
            l = int(len(X)/batch_size)*batch_size
            residuals_list[:l,:] = self.encode_batch(X[:l], Y[:l], batch_size, final=False)
            residuals_list[l:,:] = self.encode_batch(X[l:],Y[l:],1,final=True)
        elif len(X) % batch_size == 0:
            l  = len(X) - len(X) // batch_size
            if l != 0:
                residuals_list[:l,:] = self.encode_batch(X[:l], Y[:l], batch_size - 1, final=False)
            residuals_list[l:,:] = self.encode_batch(X[l:],Y[l:],1,final=True)
        else:
            raise ValueError
        print("Done")

        self.compute_compression_ratio()

        # save the residuals result
        print("Saving the results")
        load_list_to_csv(residuals_list.reshape(-1),compressed_file_name+'residuals_list.csv',self.predictor.args.data_flag,self.predictor.args.decimals)

        return 
 


    def encode_batch(self,X,Y,parallels,final=False):
        
        assert len(X) % parallels == 0
        num_iters = len(X) // parallels    
        ind = np.array(range(parallels))*num_iters
        residuals = np.zeros_like(Y)

        # store first K symbols(first time_steps) in each stream to temp_file
        f = [self.temp_file_prefix+str(final)+str(i)+'.npy' for i in range(parallels)]
        for i in range(parallels):
            np.save(f[i],X[ind[i],:self.timesteps])

        # main
        if not final:
            assert num_iters > self.timesteps//self.slide, f'iters:{num_iters} < timesteps: {self.timesteps/self.slide}'
            iters = num_iters - self.timesteps//self.slide
        else:
            iters = num_iters
        for j in tqdm(range(iters)):
            # Create Batches
            if not final:
                if self.predictor.args.data_flag == 1:
                    bx = torch.from_numpy(X[ind,:]).long()
                else:
                    bx = torch.from_numpy(X[ind,:]).float()
                by = Y[ind,:]
            else:
                if self.predictor.args.data_flag == 1:
                    bx = torch.from_numpy(X[j:j+1,:]).long()
                else:
                    bx = torch.from_numpy(X[j:j+1,:]).float()
                by = Y[j:j+1,:]

            preds = self.predictor.model_predict(bx)
            if not final:
                if self.predictor.args.data_flag == 1:
                    residuals[j:j+1,:] = by - preds
                else:
                    residuals[ind,:] = np.around(by - preds,decimals = self.predictor.args.decimals)
            else:
                if self.predictor.args.data_flag == 1:
                    residuals[j:j+1,:] = by - preds
                else:
                    residuals[j:j+1,:] = np.around(by - preds, decimals = self.predictor.args.decimals)


            ind = ind + 1

        return residuals
  


    def compute_compression_ratio(self):
        
        uncompressed_size = self.data.nbytes * 8
        metrics = {}
        metrics['File_bits'] = uncompressed_size
        with open(self.compressed_file_name+'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)



class Decoder:
    def __init__(self, mypredictor:Predictor):
        self.predictor = mypredictor      


    def decode(self,compressed_file_name: str):
        self.compressed_file_name = compressed_file_name

        # params
        params_name = compressed_file_name + 'params'
        with open(params_name,'r') as f:
            params = json.loads(f.read())
        batch_size = params['bs']
        self.timesteps = params['timesteps']
        len_series = params['len_series']
        self.slide = params['slide']
        if (len_series - self.timesteps) % self.slide == 0:
            len_x = (len_series - self.timesteps) // self.slide
        else:
            len_x = (len_series - self.timesteps) // self.slide + 1
        rem = len_x * self.slide - (len_series - self.timesteps)

        # load ranks
        residuals_list = read_list_from_csv(compressed_file_name+'residuals_list.csv')
        residuals_list = residuals_list.reshape(-1,self.slide)

        temp_dir = self.compressed_file_name + 'temp'
        self.temp_file_prefix = temp_dir + "/compressed"


        # decompress
        print("Decompressing...")
        tokens_full = np.zeros(len_series+rem,dtype=np.uint8).astype('int')

        if len_x % batch_size > 0:
            l = int(len_x/batch_size)*batch_size
            tokens_full[:l*self.slide] = self.decode_batch(l, batch_size,residuals_list[:l,:])
            tokens_full[l*self.slide:] = self.decode_batch(len_x-l+self.timesteps//self.slide, 1, residuals_list[l:,:],final = True)
        elif len_x % batch_size == 0:
            l  = len_x - len_x // batch_size
            if l != 0:
                tokens_full[:l*self.slide] = self.decode_batch(l, batch_size-1,residuals_list[:l,:])
            tokens_full[l*self.slide:] = self.decode_batch(len_x-l+self.timesteps//self.slide, 1, residuals_list[l:,:], final = True)
        else:
            raise ValueError
        
        if rem > 0 :
            tokens_full = tokens_full[:-rem]
        print("Done")


        # save decoced_results
        load_list_to_csv(tokens_full,compressed_file_name+'decompress.csv')


        return tokens_full

    

    def decode_batch(self,len_series, parallels, residuals_list,final=False):
        assert len_series % parallels == 0
        num_iters = len_series // parallels
        ind = np.array(range(parallels))*num_iters
        series_2d = np.zeros((parallels,num_iters * self.slide), dtype = np.uint8).astype('int')

        # Get first K symbols(first time_steps) in each stream from temp file
        f = [self.temp_file_prefix+str(final)+str(i)+'.npy' for i in range(parallels)]
        for i in range(parallels):
            series_2d[i,:self.timesteps] = np.load(f[i])


        # main
        j = 0
        for tj in tqdm(range(num_iters - self.timesteps // self.slide)):
            # Create Batch
            bx = torch.from_numpy(series_2d[:,j:j+self.timesteps]).long()
            preds = self.predictor.model_predict(bx)
            grounds = preds + residuals_list[ind,:]
            for it in range(self.slide):
                series_2d[:,j+self.timesteps+it] = grounds[:,it]

            ind = ind+1
            j = j + self.slide

        
        return series_2d.reshape(-1)


    def verify_text(self,data_series,decoded_data):

        if np.array_equal(data_series, decoded_data):
            print(f'Successful decoding')
        else:
            print("********!!!!! Error !!!!!*********")
        

