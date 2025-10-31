import torch
import os
import time
from transformers import AutoModel, AutoTokenizer
from predictors.Predictor import Predictor
# from Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
from utils import post_process_scores

### acquir transformers==4.41.2

class Chatglm_predictor(Predictor):

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        start_time = time.time()

        # 加载ChatGLM专用tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            '../checkpoints/Chatglm3-6b-base',
            trust_remote_code=True,
        )
        

        # 加载模型
        if self.args.use_multi_gpu:
            model = AutoModel.from_pretrained(
                '../checkpoints/Chatglm3-6b-base',
                trust_remote_code=True,
            )
            model = DDP(model.to(self.device), 
                      device_ids=[self.local_rank],
                      output_device=self.local_rank)
        else:
            model = AutoModel.from_pretrained(
                '../checkpoints/Chatglm3-6b-base',
                trust_remote_code=True
            ).to(self.device)
        
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = model.config.vocab_size
        # logits.shape[-1] = self.model.config.vocab_size = 65024 
        # self.tokenizer.vocab_size = 64798, 
        # For this model, tokenizer.vocab_size != model.config.vocab_size

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 


    def model_predict(self, input_tokens):

        # input_tokens = self.tokenizer(
        #     input_str,
        #     return_tensors="pt",
        #     add_special_tokens = False
        # )['input_ids']

        input_tokens = input_tokens.to(self.device)

        generate_kwargs = {
            "input_ids": input_tokens,
            "return_dict_in_generate":True,
            "output_scores":True,
            "max_new_tokens":self.args.pred_len,
            "do_sample":False,
            # "top_k":50,
            # "top_p":0.95,
            # "temperature":0.3,
            "renormalize_logits":True,
            # "bad_words_ids": [[token_id] for token_id in range(self.tokenizer.vocab_size, self.model.config.vocab_size)]
            "use_cache": True,        # 启用KV缓存
            "output_hidden_states": False  # 关闭不需要的输出
        }
        

        with torch.no_grad():
            if self.args.use_multi_gpu:
                outputs = self.model.module.generate(**generate_kwargs)
            else:
                outputs = self.model.generate(**generate_kwargs)

        # print(outputs['sequences'])
        # self.model_detokenize(outputs['sequences'][0])

        # print(outputs['scores'][0])
        # probs = torch.softmax(outputs['scores'][0].to(torch.float64), dim=-1).sum().items()
        # when float32, sum of probs != 1


        scores = outputs['scores']

        converted_scores = [score.to(torch.float64) for score in scores]
        final_probs = post_process_scores(converted_scores)

        return final_probs
            

    def model_tokenize(self,input_str:str):
        return self.tokenizer(input_str, add_special_tokens = False)['input_ids']
        
    
    def model_detokenize(self,input_tokens:list):
        decoded_text =  self.tokenizer.decode(input_tokens,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGLM')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='ChatGLM3')

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = Chatglm_predictor(args)
    tokens = predictor.model_tokenize("Hello! I am ")
    print(tokens)
    predictor.model_predict(torch.tensor([tokens]))

 