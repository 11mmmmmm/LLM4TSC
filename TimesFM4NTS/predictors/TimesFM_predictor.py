from predictors.Predictor import Predictor
import timesfm
import numpy as np
import torch
import time


class TimesFM_predictor(Predictor):
    def __init__(self,args): 

        # model parameters
        self.ckpt_path = '../checkpoints/torch_model.ckpt'
        self.num_layers = 50
        self.use_positional_embedding = False
        self.context_len = 2048
        self.freq = 0
        # 0: high frequency such as daily, 1: weekly and monthly, 
        # 2: low frequency, short horizon time series. 

        super().__init__(args)


    def _build_model(self):
        
        start_time = time.time()

        if self.args.use_multi_gpu:
            self.args.use_multi_gpu = False
        
        self.args.device = self.device
        # torch.cuda.set_device(self.device)

        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=self.args.win_size,  # 32
                horizon_len=self.args.pred_len,          # 128
                num_layers= self.num_layers ,
                use_positional_embedding= self.use_positional_embedding ,
                context_len= self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(path=self.ckpt_path)
        )
        # model = model.to(self.device)
            
        self.model = model

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        
        return


    def model_predict(self,input_times):
        # print(input_times)

        # transform
        freqs = torch.tensor([self.freq] * len(input_times))

        # predict
        outputs,_ =  self.model.forecast(input_times,freqs)

        # print(outputs)
        if self.args.data_flag == 1:
            outputs = np.array(outputs,dtype = 'int')
        else:
            outputs = np.array(outputs,dtype = 'float')
        # assert input_times == 0

        return outputs

    