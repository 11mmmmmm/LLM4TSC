import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from predictors.Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np
from utils import post_process_scores


# 所有 GPT-2 变种的标准上下文长度均为 1024 tokens
class GPT2_medium_process_predictor(Predictor):  

    def __init__(self, args):
        super().__init__(args)


    def _build_model(self):
        start_time = time.time()

        # ========================== 关键修改 1/3：更换模型名称和Tokenizer ==========================
        # 使用 GPT-2 的官方名称（如 'gpt2', 'gpt2-medium' 等）
        model_name = "../checkpoints/gpt2-medium"   
        
        # 加载 Tokenizer（注意 GPT-2 默认没有 pad_token，需要手动设置）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
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
        print(self.vocab_size)
        # self.good_tokens = [15,16,17,18,19,20,21,22,23,24,11,12,13]
        self.good_tokens = [657, 352, 362, 513, 604, 642, 718, 767, 807, 860, 837, 532, 764]
        
        # 变化一
        sort_good_tokens = sorted(self.good_tokens)
        sort_token_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        self.vocab_size = 13
        self.tokenstoseq = dict(zip(sort_good_tokens,sort_token_ids))
        self.seqtotokens = dict(zip(sort_token_ids,sort_good_tokens))
        self.enfunc = lambda x:[self.tokenstoseq[i] for i in x]
        self.defunc = lambda x:[self.seqtotokens[i] for i in x]
        print(self.tokenstoseq)

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

        final_probs = self.post_process_good_scores(generate_ids['scores'])
        return final_probs

    def model_tokenize(self, input_str):

        # input_tokens = self.tokenizer(['0','1','2','3','4','5','6','7','8','9',',','-','.'], add_special_tokens=False)['input_ids']
        # [[15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [11], [12], [13]]
        # input_tokens = self.tokenizer([' 0',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9',' ,',' -',' .'], add_special_tokens=False)['input_ids']
        # [[657], [352], [362], [513], [604], [642], [718], [767], [807], [860], [837], [532], [764]]
        # assert input_str == 0

        # chartoid = {'0':15,'1':16,'2':17,'3':18,'4':19,'5':20,'6':21,'7':22,'8':23,'9':24,',':11,'-':12,'.':13}
        chartoid = {'0':657,'1':352,'2':362,'3':513,'4':604,'5':642,'6':718,'7':767,'8':807,'9':860,',':837,'-':532,'.':764}
        
        dict_encode = lambda s:[chartoid[c] for c in s]
        res_tokens = dict_encode(input_str)
        print(res_tokens[:100])
        return res_tokens

        
    def model_detokenize(self, input_tokens):
        # idtochar = {15:'0',16:'1',17:'2',18:'3',19:'4',20:'5',21:'6',22:'7',23:'8',24:'9',11:',',12:'-',13:"."}
        idtochar = {657:'0',352:'1',362:'2',513:'3',604:'4',642:'5',718:'6',767:'7',807:'8',860:'9',837:',',532:'-',764:"."}

        dict_decode = lambda s:[idtochar[c] for c in s]
        res_str = dict_decode(input_tokens)
        return "".join(res_str)

    # 变化二
    def post_process_good_scores(self,generated_output):
        processed_probs = []
        
        # 遍历每个生成步骤的原始 logits
        for step_scores in generated_output:
            # 将 logits 转换为概率
            step_probs = torch.softmax(step_scores, dim=-1).detach().cpu()
            
            # 创建概率归零掩码
            # prob_mask = torch.ones_like(step_probs)
            # prob_mask[:, self.good_tokens] = 0  # 标记需要保留的位置
            
            # 执行归零操作
            zeroed_probs = step_probs[:,sorted(self.good_tokens)]
            # zeroed_probs = step_probs * (1 - prob_mask)
            # print(zeroed_probs.sum(dim=-1, keepdims=True))
            
            # 重新归一化（仅保留 good_tokens 的概率）
            renormalized_probs = zeroed_probs / zeroed_probs.sum(dim=-1, keepdims=True)
            # ind = self.good_tokens
            # renormalized_probs = renormalized_probs[:,ind]
            
            processed_probs.append(renormalized_probs.numpy())

        return np.array(processed_probs)