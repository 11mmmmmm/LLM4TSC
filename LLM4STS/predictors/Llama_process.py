import torch
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
from predictors.Predictor import Predictor
# from Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np


class Llama_process_predictor(Predictor):

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
        self.good_tokens = [29896, 29900, 29906, 29929, 29941, 29945, 29946, 29947, 29953, 29955,1919,6653,29889]


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



    def model_predict(self,input_tokens):

        # input_tokens = self.tokenizer(
        #     input_str,    # input_str 为一个batch数据，使用list表示
        #     return_tensors="pt",
        #     padding=True
        # )['input_ids']
        
        # num_input_ids = input_tokens.shape[1]
        input_tokens = input_tokens.to(self.device)

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

        final_probs = self.post_process_good_scores(generate_ids['scores'])
        # # 验证结果（示例检查最后一个生成步骤）
        # last_step_probs = final_probs[-1]
        # print("Bad tokens sum:", last_step_probs[:, bad_tokens].sum().item())  # 应输出 0.0
        # print("Good tokens sum:", last_step_probs[:, good_tokens].sum().item())  # 应输出 1.0

        return final_probs  # scores: timesteps * batch_size * vocab_size


    def model_tokenize(self,input_str):
        # input_tokens = self.tokenizer(['0','1','2','3','4','5','6','7','8','9',',','-','.'], add_special_tokens=False)['input_ids']
        # [[29871, 29900], [29871, 29896], [29871, 29906], [29871, 29941], [29871, 29946], [29871, 29945], [29871, 29953], [29871, 29955], [29871, 29947], [29871, 29929], [1919], [448], [869]]
        # 首位是数字时前面会添加29871
        # input_tokens = self.tokenizer([' 0',' 1',' 2',' 3',' 4',' 5',' 6',' 7',' 8',' 9',' ,',' -',' .'], add_special_tokens=False)['input_ids']
        # [[259, 29900], [259, 29896], [259, 29906], [259, 29941], [259, 29946], [259, 29945], [259, 29953], [259, 29955], [259, 29947], [259, 29929], [29871, 1919], [29871, 448], [29871, 869]]
        # print(self.tokenizer("2,3,45,-6", add_special_tokens=False)['input_ids'])
        # print(self.tokenizer.decode([29871, 29900,29871,29906],skip_special_tokens=True))
        # assert input_str == 0

        chartoid = {'0':29900,'1':29896,'2':29906,'3':29941,'4':29946,'5':29945,'6':29953,'7':29955,'8':29947,'9':29929,',':1919,'-':6653,'.':29889}
        
        dict_encode = lambda s:[chartoid[c] for c in s]
        res_tokens = dict_encode(input_str)
        print(res_tokens[:100])
        return res_tokens

        # input_str = ",0,123456789,"
        # input_tokens = self.tokenizer(input_str, add_special_tokens = False)['input_ids'][1:]
        # # input_tokens = self.tokenizer(input_str, add_special_tokens = False)['input_ids']
        # print(input_tokens[:100])
        # return input_tokens

    
    def model_detokenize(self,input_tokens):
        idtochar = {29900:'0',29896:'1',29906:'2',29941:'3',29946:'4',29945:'5',29953:'6',29955:'7',29947:'8',29929:'9',1919:',',6653:'-',29889:"."}

        dict_decode = lambda s:[idtochar[c] for c in s]
        res_str = dict_decode(input_tokens)
        return "".join(res_str)

        # return self.tokenizer.decode(input_tokens,skip_special_tokens=True)


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM')
    
    # base
    parser.add_argument('--model_name', type=str,default='Llama')
    parser.add_argument('--model_dir', type=str,default='../../checkpoints/Llama7B-hf')
    parser.add_argument('--pred_len', type=int, default=2)
    # gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=3, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = Llama_process_predictor(args)
    # tokens = predictor.model_tokenize("China is a beautiful country, Welcome to ")
    predictor.model_init()
    tokens = predictor.model_tokenize("1,2,3,4,5,6,7")
    print(tokens)
    predictor.model_predict(torch.tensor([tokens]))
    