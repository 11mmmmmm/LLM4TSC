from predictors.Predictor import Predictor
from transformers import AutoModelForCausalLM
import numpy as np
import torch
import time


class Timerxl_predictor(Predictor):
    def __init__(self,args): 

        super().__init__(args)


    def _build_model(self):
        
        start_time = time.time()

        # get model
        if self.args.use_multi_gpu:
            raise IOError
        else: 
            self.args.device = self.device
            model = AutoModelForCausalLM.from_pretrained('../checkpoints/timer-base-84m', 
                                                         trust_remote_code=True).to(self.device)
        
        model.eval()

        self.model = model

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 


    def model_predict(self,input_times):

        # transform
        # print(input_times)
        input_times = input_times.float().to(self.device)

        # predict
        outputs = self.model.generate(input_times, 
                                      max_new_tokens=self.args.pred_len)

        if self.args.data_flag == 1:
            outputs = torch.round(outputs).detach().cpu().numpy()
            outputs = np.array(outputs,dtype = 'int')
        else:
            outputs = outputs.detach().cpu().numpy()
            outputs = np.array(outputs,dtype = 'float')

        # print(outputs)
        # assert input_times == 0

        return outputs

    