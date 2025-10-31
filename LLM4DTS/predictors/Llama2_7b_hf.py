import torch
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
from utils import post_process_scores

# 上下文长度不超过4096 tokens
class Llama2_7b_hf_predictor(Predictor):

    def __init__(self,args):

        self.DEFAULT_EOS_TOKEN = "</s>"
        self.DEFAULT_BOS_TOKEN = "<s>"
        self.DEFAULT_UNK_TOKEN = "<unk>"
        super().__init__(args)


    def _build_model(self):

        start_time = time.time()

        # get tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            '../checkpoints/Llama7B-hf',
            use_fast=False,
            padding_side='left'
        )
        special_tokens_dict = dict()
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = self.DEFAULT_EOS_TOKEN  # end
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = self.DEFAULT_BOS_TOKEN  # begin
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = self.DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = tokenizer.eos_token    # padding


        # get model
        if self.args.use_multi_gpu:
            model = LlamaForCausalLM.from_pretrained('../checkpoints/Llama7B-hf', 
                                                     torch_dtype=torch.float16)
            model = DDP(model.to(self.device),
                         device_ids=[self.local_rank],
                         output_device=self.local_rank, 
                         find_unused_parameters=True)
        else: 
            self.args.device = self.device
            model = LlamaForCausalLM.from_pretrained(
                '../checkpoints/Llama7B-hf',
                # device_map="auto",  
                torch_dtype=torch.float16,
            ).to(self.device)
        
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 



    def model_predict(self,input_tokens):

        # input_tokens = self.tokenizer(
        #     input_str,    # input_str 为一个batch数据，使用list表示
        #     return_tensors="pt",
        #     padding=True
        # )['input_ids']

        input_tokens = input_tokens.to(self.device)

        generate_input = {
            "input_ids":input_tokens,
            "return_dict_in_generate":True,
            "output_scores":True,
            "max_new_tokens":self.args.pred_len,
            "do_sample":False,
            "use_cache": True,  # 启用缓存提升生成速度
            "output_hidden_states": False,  # 关闭不需要的输出
            # "top_k":50,
            # "top_p":0.95,
            # "temperature":0.3,
            "renormalize_logits":True,
        }

        with torch.no_grad():
            if self.args.use_multi_gpu:
                generate_ids = self.model.module.generate(**generate_input)
            else:
                generate_ids = self.model.generate(**generate_input)

        # self.tokenizer.batch_decode(
        #     generate_ids[:, num_input_ids:],
        #     skip_special_tokens=True, 
        #     clean_up_tokenization_spaces=False
        # )

        final_probs = post_process_scores(generate_ids['scores'])

        return final_probs  # scores: timesteps * batch_size * vocab_size


    def model_tokenize(self,input_str):
        return self.tokenizer(input_str, add_special_tokens = False)['input_ids']
        
    
    def model_detokenize(self,input_tokens):
        return self.tokenizer.decode(input_tokens,skip_special_tokens=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM')
    
    # base
    parser.add_argument('--model_name', type=str,default='Llama')
    parser.add_argument('--model_dir', type=str,default='../../checkpoints/Llama7B-hf')

    # gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    mypredictor = Llama2_7b_hf_predictor(args)
    mypredictor.model_predict(["Hello World!"])

