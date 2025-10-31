from abc import ABC, abstractmethod
import os
import torch



class Predictor(ABC):
    def __init__(self,args):
        self.args = args
        self.device = self._acquire_device()
        self._build_model()


    def _build_model(self):
        raise NotImplementedError
        return None    


    def _acquire_device(self):

        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = torch.device(f'cuda:{self.local_rank}')
                print(f'[Multi-GPU] Using GPU: {device}')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'[Single-GPU] Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device 

