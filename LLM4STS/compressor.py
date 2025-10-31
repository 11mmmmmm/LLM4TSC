import os
import torch
import numpy as np
from utils import *
from encoders import *
from predictors import Predictor
from tqdm import tqdm
import shutil

 
class Encoder:
    def __init__(self,mypredictor:Predictor):
        self.predictor = mypredictor


    def encode( self, 
                text_input, 
                timesteps=64,
                slide = 1,
                compressed_file_name: str = 'LLMzip',
                batch_size = 32
            ):
            

        self.compressed_file_name = compressed_file_name
        self.timesteps = timesteps
        self.slide = slide
        self.vocab_size = self.predictor.vocab_size

        # Tokenize
        print("Tokenizing...")
        tokens_full = np.array(self.predictor.model_tokenize(text_input))
        print("Done")
        
        len_series = len(tokens_full)
        if (len_series - timesteps) % slide == 0:
            ind = (len_series - timesteps) // slide
        else:
            ind = (len_series - timesteps) // slide + 1
        rem = ind * slide - (len_series - timesteps)
        # 不要全部填充为0！！！！否则tokenstoseq会报错
        tokens_full = np.concatenate((tokens_full,np.full(rem,self.predictor.good_tokens[-1],dtype=int)))
        tokens_data = strided_app(tokens_full,timesteps,slide)
        X = tokens_data[:,:-slide]
        Y = tokens_data[:,-slide:]

        # tokens_data = strided_app(tokens_full, timesteps+1, 1)
        # X = tokens_data[:, :-1]
        # Y = tokens_data[:, -1]

        # params
        params_name = compressed_file_name+"params"
        params = {}
        params['len_series'] = len_series
        params['bs'] = batch_size
        params['timesteps'] = timesteps
        params['slide'] = slide
        
        with open(params_name,'w') as f:
            json.dump(params, f, indent=4)

        # create temp dir
        temp_dir = self.compressed_file_name + 'temp'
        if os.path.exists(temp_dir):
            os.system("rm -r {}".format(temp_dir))
        self.temp_file_prefix = temp_dir + "/compressed"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # compress
        # Final:False, 对于每个batch，第一个timesteps使用平均概率压缩，最后一个timesteps对应的数值无需预测
        # Final:True, 只有一个batch也是最后一组预测，第一个timesteps使用平均概率压缩
        print("Compressing...")
        
        ranks_list = np.zeros_like(Y,dtype=int)
        freqs_list = np.zeros_like(Y,dtype=np.float32)
        if len(X) % batch_size > 0:
            l = int(len(X)/batch_size)*batch_size
            ranks_list[:l,:],freqs_list[:l,:] = self.encode_batch(X[:l], Y[:l], batch_size, final=False)
            ranks_list[l:,:],freqs_list[l:,:] = self.encode_batch(X[l:],Y[l:],1,final=True)
        elif len(X) % batch_size == 0:
            l  = len(X) - len(X) // batch_size
            if l != 0:
                ranks_list[:l,:],freqs_list[:l,:] = self.encode_batch(X[:l], Y[:l], batch_size - 1, final=False)
            ranks_list[l:,:],freqs_list[l:,:] = self.encode_batch(X[l:],Y[l:],1,final=True)
        else:
            raise ValueError
        print("Done")
        

        # combine files into one file
        f = open(compressed_file_name+'compressed','wb')
        num_batch = batch_size if len(X) % batch_size > 0 else batch_size-1
        for i in range(num_batch):
            f_in = open(self.temp_file_prefix+str(False)+'.'+str(i),'rb')
            byte_str = f_in.read()
            byte_str_len = len(byte_str)
            var_int_encode(byte_str_len, f)
            f.write(byte_str)
            f_in.close()
        f_in = open(self.temp_file_prefix+str(True)+'.0','rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
        f.close()
            
        shutil.rmtree(self.compressed_file_name + 'temp')  
        
        self.compute_compression_ratio()

        # save the ranks result
        load_np_to_txt(ranks_list.reshape(-1),compressed_file_name+'ranks')
        load_np_to_txt(freqs_list.reshape(-1),compressed_file_name+'probs')
    


    def encode_batch(self,X,Y,parallels,final=False):
        
        assert len(X) % parallels == 0
        num_iters = len(X) // parallels    
        ind = np.array(range(parallels))*num_iters
        ranks,freqs = np.zeros_like(Y),np.zeros_like(Y,dtype=np.float32)
        # gens = []

        f = [open(self.temp_file_prefix+str(final)+'.'+str(i),'wb') for i in range(parallels)]
        bitout = [BitOutputStream(f[i]) for i in range(parallels)]
        enc = [ArithmeticEncoder(32, bitout[i]) for i in range(parallels)]

        # Encode first K symbols(first time_steps) in each stream with uniform probabilities
        prob = np.ones(self.vocab_size)/self.vocab_size
        cumul = np.zeros(self.vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        for i in range(parallels):
            for j in range(self.timesteps):
                # 变化1
                enc[i].write(cumul,self.predictor.tokenstoseq[X[ind[i],j]])
 

        # main
        cumul = np.zeros((parallels, self.vocab_size+1), dtype = np.uint64)
        if not final:
            assert num_iters > self.timesteps//self.slide, f'iters:{num_iters} < timesteps: {self.timesteps/self.slide}'
            iters = num_iters - self.timesteps//self.slide
        else:
            iters = num_iters
        for j in tqdm(range(iters)):
            # Create Batches
            if not final:
                bx = torch.from_numpy(X[ind,:]).long()
                by =torch.from_numpy(Y[ind,:])
            else:
                bx = torch.from_numpy(X[j:j+1,:]).long()
                by = torch.from_numpy(Y[j:j+1,:])


            probs = self.predictor.model_predict(bx)
            # gens += preds
            # 变化2
            for it in range(self.slide):
                # print(by[:,it])
                rank = gen_rank(probs[it],next_token=torch.tensor(self.predictor.enfunc(by[:,it].tolist()))).cpu().numpy()
                if not final:
                    ranks[ind,it] = rank
                    freqs[ind,it] = probs[it,np.arange(parallels),self.predictor.enfunc(Y[ind,it])]
                else:
                    ranks[j:j+1,it] = rank
                    freqs[j:j+1,it] = probs[it,np.arange(parallels),self.predictor.enfunc(Y[j:j+1,it])]

                cumul[:,1:] = np.cumsum(probs[it]*10000000 + 1, axis = 1)
                
                # Encode with Arithmetic Encoder
                for i in range(parallels):
                    # 变化3
                    enc[i].write(cumul[i,:], self.predictor.tokenstoseq[Y[ind[i],it]])

            ind = ind + 1


        # close files
        for i in range(parallels):
            enc[i].finish()
            bitout[i].close()
            f[i].close()


        # if self.predictor.args.text_type == 0:
        #     genstr = self.predictor.model_detokenize(list(X[0]))
        #     genstr += "".join(gens)
        #     with open(self.compressed_file_name+'generates','w') as f:
        #         f.write(genstr)

        return ranks,freqs


    def compute_compression_ratio(self):
        metrics = {}
        metrics['File_bits'] = cal_file_bits(self.predictor.args.text_file)
        metrics['AC_bits'] = cal_file_bits(self.compressed_file_name+'compressed')
        metrics['AC_ratio'] = metrics['AC_bits']/metrics['File_bits']
        with open(self.compressed_file_name+'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
                    

  

class Decoder:
    def __init__(self, mypredictor:Predictor):
        self.predictor = mypredictor

        
    def decode(
        self,
        compressed_file_name: str
        ):

        self.compressed_file_name = compressed_file_name

        # params
        params_name = compressed_file_name + 'params'
        with open(params_name,'r') as f:
            params = json.loads(f.read())
        batch_size = params['bs']
        self.timesteps = params['timesteps']
        len_series = params['len_series']
        self.slide = params['slide']
        self.vocab_size = self.predictor.vocab_size

        if (len_series - self.timesteps) % self.slide == 0:
            len_x = (len_series - self.timesteps) // self.slide
        else:
            len_x = (len_series - self.timesteps) // self.slide + 1
        rem = len_x * self.slide - (len_series - self.timesteps)


        # create temp dir
        temp_dir = self.compressed_file_name + 'temp'
        if os.path.exists(temp_dir):
            os.system("rm -r {}".format(temp_dir))
        self.temp_file_prefix = temp_dir + "/compressed"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Break into multiple streams
        f = open(compressed_file_name+'compressed','rb')
        num_batch = batch_size if len_x % batch_size > 0 else batch_size-1
        for i in range(num_batch):
            f_out = open(self.temp_file_prefix+str(False)+'.'+str(i),'wb')
            byte_str_len = var_int_decode(f)
            byte_str = f.read(byte_str_len)
            f_out.write(byte_str)
            f_out.close()
        f_out = open(self.temp_file_prefix+str(True)+'.0','wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
        f.close()

        # decompress
        print("Decompressing...")
        tokens_full = np.zeros(len_series+rem,dtype=np.uint8).astype('int')
        
        if len_x % batch_size > 0:
            l = int(len_x/batch_size)*batch_size
            tokens_full[:l*self.slide] = self.decode_batch(l, batch_size)
            tokens_full[l*self.slide:] = self.decode_batch(len_x-l+self.timesteps//self.slide, 1, final = True)
        elif len_x % batch_size == 0:
            l  = len_x - len_x // batch_size
            if l != 0:
                tokens_full[:l*self.slide] = self.decode_batch(l, batch_size-1)
            tokens_full[l*self.slide:] = self.decode_batch(len_x-l+self.timesteps//self.slide, 1, final = True)
        else:
            raise ValueError
        
        if rem > 0 :
            tokens_full = tokens_full[:-rem]
        print("Done")


        # Detokenize
        print("Detokenizing...")
        decoded_text = self.predictor.model_detokenize(list(tokens_full))
        with open(compressed_file_name+'decompress','w') as f:            
            f.write(decoded_text)
        print("Done")


        shutil.rmtree(self.compressed_file_name + 'temp') 


    def decode_batch(self,len_series, parallels, final=False):
        
        assert len_series % parallels == 0
        num_iters = len_series // parallels
        series_2d = np.zeros((parallels,num_iters * self.slide), dtype = np.uint8).astype('int')

        f = [open(self.temp_file_prefix+str(final)+'.'+str(i),'rb') for i in range(parallels)]
        bitin = [BitInputStream(f[i]) for i in range(parallels)]
        dec = [ArithmeticDecoder(32, bitin[i]) for i in range(parallels)]

        # Decode first K symbols(first time_steps) in each stream with uniform probabilities
        prob = np.ones(self.vocab_size)/self.vocab_size
        cumul = np.zeros(self.vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        for i in range(parallels):
            for j in range(self.timesteps):
                # 变化1
                series_2d[i,j] = self.predictor.seqtotokens[dec[i].read(cumul, self.vocab_size)]


        # main
        cumul = np.zeros((parallels, self.vocab_size+1), dtype = np.uint64)
        j = 0
        for tj in tqdm(range(num_iters - self.timesteps // self.slide)):
            # Create Batch
            bx = torch.from_numpy(series_2d[:,j:j+self.timesteps]).long()
            probs = self.predictor.model_predict(bx)
            for it in range(self.slide):
                cumul[:,1:] = np.cumsum(probs[it]*10000000 + 1, axis = 1)
                # Decode with Arithmetic Encoder
                for i in range(parallels):
                    # 变化2
                    series_2d[i,j+self.timesteps+it] = self.predictor.seqtotokens[dec[i].read(cumul[i,:], self.vocab_size)]
            j = j + self.slide


        # close files
        for i in range(parallels):
            bitin[i].close()
            f[i].close()
        
        return series_2d.reshape(-1)
        

    def decode_ranks(
        self,
        compressed_file_name: str
        ):

        self.compressed_file_name = compressed_file_name

        # params
        params_name = compressed_file_name + 'params'
        with open(params_name,'r') as f:
            params = json.loads(f.read())
        batch_size = params['bs']
        self.timesteps = params['timesteps']
        len_series = params['len_series']
        self.slide = params['slide']
        self.vocab_size = self.predictor.vocab_size
        if (len_series - self.timesteps) % self.slide == 0:
            len_x = (len_series - self.timesteps) // self.slide
        else:
            len_x = (len_series - self.timesteps) // self.slide + 1
        rem = len_x * self.slide - (len_series - self.timesteps)


        # load ranks
        ranks_list = read_np_from_txt(compressed_file_name+'ranks')
        ranks_list = ranks_list.reshape(-1,self.slide)


        # create temp dir
        temp_dir = self.compressed_file_name + 'temp'
        if os.path.exists(temp_dir):
            os.system("rm -r {}".format(temp_dir))
        self.temp_file_prefix = temp_dir + "/compressed"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Break into multiple streams
        f = open(compressed_file_name+'compressed','rb')
        num_batch = batch_size if len_x % batch_size > 0 else batch_size-1
        for i in range(num_batch):
            f_out = open(self.temp_file_prefix+str(False)+'.'+str(i),'wb')
            byte_str_len = var_int_decode(f)
            byte_str = f.read(byte_str_len)
            f_out.write(byte_str)
            f_out.close()
        f_out = open(self.temp_file_prefix+str(True)+'.0','wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
        f.close()

        # decompress
        print("Decompressing...")
        tokens_full = np.zeros(len_series+rem,dtype=np.uint8).astype('int')


        if len_x % batch_size > 0:
            l = int(len_x/batch_size)*batch_size
            tokens_full[:l*self.slide] = self.decode_ranks_batch(l, batch_size,ranks_list[:l,:])
            tokens_full[l*self.slide:] = self.decode_ranks_batch(len_x-l+self.timesteps//self.slide, 1, ranks_list[l:,:],final = True)
        elif len_x % batch_size == 0:
            l  = len_x - len_x // batch_size
            if l != 0:
                tokens_full[:l*self.slide] = self.decode_ranks_batch(l, batch_size-1,ranks_list[:l,:])
            tokens_full[l*self.slide:] = self.decode_ranks_batch(len_x-l+self.timesteps//self.slide, 1, ranks_list[l:,:], final = True)
        else:
            raise ValueError
        
        if rem > 0 :
            tokens_full = tokens_full[:-rem]
        print("Done")


        # Detokenize
        print("Detokenizing...")
        decoded_text = self.predictor.model_detokenize(list(tokens_full))
        with open(compressed_file_name+'decompress','w') as f:            
            f.write(decoded_text)
        print("Done")


        shutil.rmtree(self.compressed_file_name + 'temp') 


    def decode_ranks_batch(self,len_series, parallels, ranks_list,final=False):
        
        assert len_series % parallels == 0
        num_iters = len_series // parallels
        ind = np.array(range(parallels))*num_iters
        series_2d = np.zeros((parallels,num_iters * self.slide), dtype = np.uint8).astype('int')

        f = [open(self.temp_file_prefix+str(final)+'.'+str(i),'rb') for i in range(parallels)]
        bitin = [BitInputStream(f[i]) for i in range(parallels)]
        dec = [ArithmeticDecoder(32, bitin[i]) for i in range(parallels)]

        # Decode first K symbols(first time_steps) in each stream with uniform probabilities
        prob = np.ones(self.vocab_size)/self.vocab_size
        cumul = np.zeros(self.vocab_size+1, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        for i in range(parallels):
            for j in range(self.timesteps):
                # 变化1
                series_2d[i,j] = self.predictor.seqtotokens[dec[i].read(cumul, self.vocab_size)]

        # main
        cumul = np.zeros((parallels, self.vocab_size+1), dtype = np.uint64)
        j = 0
        for tj in tqdm(range(num_iters - self.timesteps // self.slide)):
            # Create Batch
            bx = torch.from_numpy(series_2d[:,j:j+self.timesteps]).long()
            probs = self.predictor.model_predict(bx)
            for it in range(self.slide):
                # 变化2
                series_2d[:,j+self.timesteps+it] = self.predictor.defunc([gen_next_token(probs[it],ranks_list[ind,it]).tolist()])

            ind = ind+1
            j = j + self.slide

        
        return series_2d.reshape(-1)
        


