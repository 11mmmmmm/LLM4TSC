import torch
import os
import time
from transformers import XLNetTokenizer, XLNetLMHeadModel, AutoTokenizer, AutoModelForCausalLM
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np
from utils import post_process_scores


class XLNet_large_predictor(Predictor):  

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        start_time = time.time()

        # ========================== 关键修改 1/2：更换为 XLNet ==========================
        model_name = "../checkpoints/xlnet-large-cased"  # 官方模型名称（自动下载）或本地路径
        
        # 加载 XLNet 的专用分词器
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        
        # 加载 XLNet 语言模型（根据任务选择对应 Head）
        if self.args.use_multi_gpu:
            model = XLNetLMHeadModel.from_pretrained(model_name)
            model = DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank, 
                find_unused_parameters=True
            )
        else: 
            model = XLNetLMHeadModel.from_pretrained(model_name).to(self.device)
        # =================================================================================

        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        assert tokenizer.vocab_size == model.config.vocab_size
        self.vocab_size = tokenizer.vocab_size
        
        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 
    

    def model_predict(self, input_tokens):
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
        # num_input_ids = input_tokens.shape[1]
        # print(self.tokenizer.batch_decode(
        #     generate_ids['sequences'][:, num_input_ids:],
        #     skip_special_tokens=True, 
        #     clean_up_tokenization_spaces=False
        # ))
        return final_probs
    

    def model_tokenize(self, input_str):
        return self.tokenizer(input_str, add_special_tokens=False)['input_ids']
    
        
    def model_detokenize(self, input_tokens):
        """解码时需处理 XLNet 的特殊空格逻辑"""
        decoded_str = self.tokenizer.decode(
            input_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        # return decoded_str
        # XLNet 分词器可能插入特殊空格，需手动修复
        return decoded_str.replace("▁", " ").strip()  # 处理 SentencePiece 的分词符号
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Xlnet')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='Xlnet')
    parser.add_argument('--pred_len', type=int, default=2)
    
    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = XLNet_large_predictor(args)

    # tokens = predictor.model_tokenize("[Thu Jun 09 06:07:05 2005] [notice] Digest: generating secret for digest authentication ...")
    tokens = predictor.model_tokenize("China is a beautiful")
    predictor.model_predict(torch.tensor([tokens]))
    # print(tokens)
    strs = predictor.model_detokenize(tokens)
    print(strs)
    # for i in tokens:
    #     print(predictor.model_detokenize([i]))


