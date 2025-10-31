import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
# from Predictor import Predictor
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import warnings
warnings.filterwarnings('ignore')
from utils import post_process_scores

  
class DeepseekQwen_predictor(Predictor):

    def __init__(self,args):
        super().__init__(args)


    def _build_model(self):

        start_time = time.time()

        # get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            '../checkpoints/Deepseek-R1-Qwen-1.5b',    
            trust_remote_code=True , 
            )

        # get model
        if self.args.use_multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(
                '../checkpoints/Deepseek-R1-Qwen-1.5b',
                trust_remote_code=True 
                )
            model = DDP(model.to(self.device),
                         device_ids=[self.local_rank],
                         output_device=self.local_rank, 
                         find_unused_parameters=True
                         )
        else: 
            self.args.device = self.device
            model = AutoModelForCausalLM.from_pretrained(
                '../checkpoints/Deepseek-R1-Qwen-1.5b',
                trust_remote_code=True 
            ).to(self.device)
        
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        # For this model, tokenizer.vocab_size != model.config.vocab_size

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 



    def model_predict(self,input_tokens):

        input_tokens = input_tokens.to(self.device)
        attention_mask = torch.ones(input_tokens.shape,dtype=torch.long,device=self.device)

        generate_input = {
            "input_ids":input_tokens,
            "return_dict_in_generate":True,
            "output_scores":True,
            "max_new_tokens":self.args.pred_len,
            "do_sample":False,
            "attention_mask":attention_mask,
            "pad_token_id":self.tokenizer.eos_token_id,
            # "top_k":50,
            # "top_p":0.95,
            # "temperature":0.3,
            "renormalize_logits":True,
            "use_cache": True,        # 启用KV缓存
            "output_hidden_states": False  # 关闭不需要的输出
        }

        with torch.no_grad():
            if self.args.use_multi_gpu:
                generate_ids = self.model.module.generate(**generate_input)
            else:
                generate_ids = self.model.generate(**generate_input)

        # print(generate_ids['sequences'])
        # print(self.model_detokenize(generate_ids['sequences'][0]))


        final_probs = post_process_scores(generate_ids['scores'])

        return final_probs  # scores: timesteps * batch_size * vocab_size


    def model_tokenize(self,input_str):
        return self.tokenizer(input_str, add_special_tokens = False)['input_ids']
        
    
    def model_detokenize(self,input_tokens):
        return self.tokenizer.decode(input_tokens,skip_special_tokens=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGLM')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='ChatGLM3')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = DeepseekQwen_predictor(args)

    tokens = predictor.model_tokenize("China is a beautiful country, Welcome to ")
    print(tokens)
    predictor.model_predict(torch.tensor([tokens]))