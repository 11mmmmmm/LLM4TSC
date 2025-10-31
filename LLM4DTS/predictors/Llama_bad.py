import torch
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np
from utils import post_process_scores

  
class Llama_bad_predictor(Predictor):

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
        
        # num_input_ids = input_tokens.shape[1]
        input_tokens = input_tokens.to(self.device)

        good_tokens_str = list("0123456789,")
        good_tokens = [self.tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        bad_tokens = [i for i in range(len(self.tokenizer)) if i not in good_tokens]

        generate_input = {
            "input_ids":input_tokens,
            "return_dict_in_generate":True,
            "output_scores":True,
            "max_new_tokens":self.args.pred_len,
            "do_sample":False,
            "use_cache": True, 
            "output_hidden_states": False,  # 关闭不需要的输出
            # "top_k":50,
            # "top_p":0.95,
            # "temperature":0.3,
            "bad_words_ids": [[t] for t in bad_tokens],
            "renormalize_logits":True,
        }

        with torch.no_grad():
            if self.args.use_multi_gpu:
                generate_ids = self.model.module.generate(**generate_input)
            else:
                generate_ids = self.model.generate(**generate_input)


        # gen_strs = []
        # gen_strs += self.tokenizer.batch_decode(
        #     generate_ids['sequences'][:, num_input_ids:],
        #     skip_special_tokens=True, 
        #     clean_up_tokenization_spaces=False
        # )

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

