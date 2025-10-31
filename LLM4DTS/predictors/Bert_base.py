import torch
import os
import time
from transformers import BertTokenizer, AutoModelForMaskedLM
from predictors.Predictor import Predictor
# from Predictor import Predictor
from torch.nn.parallel import DistributedDataParallel as DDP 
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Bert模型由于分词无法复原，不予采用
class BERT_base_predictor(Predictor):  

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        start_time = time.time()

        model_name = "../checkpoints/bert-base-uncased"  
        model_name = "../../checkpoints/bert-base-uncased"  
        
        # 加载BERT的Tokenizer（自动包含[CLS]、[SEP]等特殊token）
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 加载Masked Language Model
        if self.args.use_multi_gpu:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model = DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank, 
                find_unused_parameters=True
            )
        else: 
            model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)

        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        assert tokenizer.vocab_size == model.config.vocab_size
        self.vocab_size = tokenizer.vocab_size
        print(self.vocab_size)
        
        print(f"{self.device}: Loaded in {time.time() - start_time:.2f} seconds")
        return 

    def model_predict(self, input_tokens):
        input_tokens = input_tokens.to(self.device)
        batch_size, seq_len = input_tokens.shape
        
        # 在输入序列末尾添加pred_len个[MASK]
        extended_tokens = torch.cat([
            input_tokens,
            torch.full((batch_size, self.args.pred_len), 
                       self.tokenizer.mask_token_id,
                       device=self.device)
        ], dim=1)

        attention_mask = (extended_tokens != self.tokenizer.pad_token_id).long().to(self.device)

        '''
        BERT 直接通过前向传播输出所有位置的 logits,用于掩码预测而非文本生成
        而无需使用生成策略(sampling/beam search), 不需要do_sample=False参数,
        '''
        
        with torch.no_grad():
            outputs = self.model(extended_tokens, attention_mask=attention_mask)
        
        # 获取MASK位置的预测logits
        # print(outputs)
        # print(outputs.logits.shape)
        mask_logits = outputs.logits[:, seq_len:, :]  # 获取所有MASK位置的预测

        # batch_size * time_steps * vocab_size
        final_probs = np.array([torch.softmax(mask_logits[:, i, :], dim=-1).detach().cpu().numpy() for i in range(self.args.pred_len)])
        # time_steps * batch_size * vocab_size
        return final_probs
    

    def model_tokenize(self, input_str):
        tmp = self.tokenizer.tokenize(input_str)
        print(tmp)
        return self.tokenizer.convert_tokens_to_ids(tmp)
        
    def model_detokenize(self, input_tokens):
        # print(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_tokens)))
        # same results 
        return self.tokenizer.decode(input_tokens,clean_up_tokenization_spaces=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT Predictor')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--pred_len', type=int, default=2)

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    predictor = BERT_base_predictor(args)

    tokens = predictor.model_tokenize("[Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK, Digest: generating secret for digest authentication ...")
    # tokens = predictor.model_tokenize("why isn't Alex's text tokenizing? The house on the left is the Smiths' house")
    # print(tokens)
    # predictor.model_predict(torch.tensor([tokens]))
    print(predictor.model_detokenize(tokens))
