import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
from utils import post_process_scores


# 所有 GPT-2 变种的标准上下文长度均为 1024 tokens
class GPT2_large_predictor(Predictor):  

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        start_time = time.time()

        # ========================== 关键修改 1/3：更换模型名称和Tokenizer ==========================
        # 使用 GPT-2 的官方名称（如 'gpt2', 'gpt2-medium' 等）
        model_name = "../checkpoints/gpt2-large"  
        
        # 加载 Tokenizer（注意 GPT-2 默认没有 pad_token，需要手动设置）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # trust_remote_code=True  # GPT-2 不需要此参数
            pad_token="<|endoftext|>"  # 显式设置 pad_token 为 eos_token
        )
        
        # 加载模型
        if self.args.use_multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank, 
                find_unused_parameters=True
            )
        else: 
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        # =======================================================================================

        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        assert tokenizer.vocab_size == model.config.vocab_size
        self.vocab_size = tokenizer.vocab_size  # GPT-2 的 tokenizer 和模型 vocab_size 一致
        
        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 


    def model_predict(self, input_tokens):

        # 将输入限制在win_size个字符，而不是win_size个tokens，注：下面仅限batch=1
        # print(input_tokens)
        input_str = self.model_detokenize(input_tokens.tolist()[0])
        # print(input_str)
        # print(len(input_str))
        input_str = input_str[-self.args.win_size:]
        # print(input_str)
        # print(len(input_str))
        # if self.args.gpt_type == 'r':
        #     input_str = " " + " ".join(list(input_str)) + " "
        input_tokens = self.model_tokenize(input_str)  
        input_tokens = torch.tensor([input_tokens]).long()
        # print(len(input_tokens[0]))
        assert input_tokens == 0   

        input_tokens = input_tokens.to(self.device)
        attention_mask = (input_tokens != self.tokenizer.pad_token_id).long().to(self.device)  # 更安全的mask生成

        # ========================== 关键修改 2/3：调整生成参数 ==========================
        generate_input = {
            "input_ids": input_tokens,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": self.args.pred_len,
            "do_sample": False,
            "attention_mask": attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,  # GPT-2 使用 eos_token 作为 padding
            "renormalize_logits": True,
            "use_cache": True,        # 启用KV缓存
            "output_hidden_states": False  # 关闭不需要的输出
        }
        # ==============================================================================
        
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
        return self.tokenizer.decode(input_tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False  # 禁止自动清理空格
                                     )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ChatGLM')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='ChatGLM3')
    parser.add_argument('--pred_len', type=int, default=2)
    
    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = GPT2_large_predictor(args)

    tokens = predictor.model_tokenize("China is a beautiful country, Welcome to ")
    print(tokens)
    predictor.model_predict(torch.tensor([tokens]))