import torch
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np
from utils import post_process_scores



class Llama2_13b_hf_predictor(Predictor):  # 修改类名

    def __init__(self, args):
        self.DEFAULT_EOS_TOKEN = "</s>"
        self.DEFAULT_BOS_TOKEN = "<s>"
        self.DEFAULT_UNK_TOKEN = "<unk>"
        super().__init__(args)

    def _build_model(self):
        start_time = time.time()

        # ==================== 关键修改 1/3：修改模型路径 ====================
        base_model_path = "../checkpoints/Llama13B"  # 13B模型路径
        
        # 加载tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            base_model_path,  # 更新路径
            use_fast=False,
            padding_side='left'
        )
        
        # 处理特殊token
        special_tokens_dict = dict()
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = self.DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = self.DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = self.DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = tokenizer.eos_token
        # ================================================================

        # ================== 关键修改 2/3：优化大模型加载方式 ==================
        load_config = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,  # 减少CPU内存占用
            "temperature":None,    # 禁用 temperature
            "top_p":None,         # 禁用 top_p
            "do_sample":False     # 确保关闭采样
        }
        
        if self.args.use_multi_gpu:
            model = LlamaForCausalLM.from_pretrained(
                base_model_path, 
                **load_config
            )
            model = DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model_path,
                **load_config
            ).to(self.device)
        # ================================================================
        
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        print(f"{self.device}: Loaded in {time.time() - start_time:.2f}s")
        return 

    def model_predict(self, input_tokens):
        input_tokens = input_tokens.to(self.device)

        # ============ 关键修改 3/3：添加大模型生成优化参数 ============
        generate_input = {
            "input_ids": input_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": self.args.pred_len,
            "do_sample": False,
            "use_cache": True,  # 启用缓存提升生成速度
            "output_hidden_states": False,  # 关闭不需要的输出
            # "top_k":50,
            # "top_p":0.95,
            # "temperature":0.3,
            "renormalize_logits": True,
        }

        with torch.no_grad():
            if self.args.use_multi_gpu:
                generate_ids = self.model.module.generate(**generate_input)
            else:
                generate_ids = self.model.generate(**generate_input)

        final_probs = post_process_scores(generate_ids['scores'])
        return final_probs


    def model_tokenize(self, input_str):
        return self.tokenizer(input_str, add_special_tokens=False)['input_ids']
    
    def model_detokenize(self, input_tokens):
        return self.tokenizer.decode(input_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM')
    
    # ============ 修改默认模型路径 ============
    parser.add_argument('--model_dir', type=str, default='../../checkpoints/Llama13B-hf')
    parser.add_argument('--pred_len', type=int, default=2)

    parser.add_argument('--model_name', type=str, default='Llama')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # 注意修改实例化类名
    predictor = Llama2_13b_hf_predictor(args)
    tokens = predictor.model_tokenize("China is a beautiful country, Welcome to ")
    print(tokens)
    predictor.model_predict(torch.tensor([tokens]))

