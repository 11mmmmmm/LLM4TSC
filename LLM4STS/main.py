import os
import sys
import time
import json
from predictors import *
import torch
from compressor import Encoder,Decoder
import warnings
import argparse
import torch.distributed as dist

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Large Language Model for DTS')
    # base
    parser.add_argument('--model_name', type=str,default='gpt2',help = 'the name of model')
    parser.add_argument('--text_file', type=str,default='../Dataset/Float/text_syndata_climate0',help='text_ZH_root')
    parser.add_argument('--win_size', type=int, default=256, help='the context window length and it cannot exceed the max seq length')
    parser.add_argument('--pred_len', type=int, default=32, help='the predicted tokens length')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--encode_decode', type=int, default=0,help='0: encode, 1: decode')
    
    # gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default="0,2,3", help='device ids of multile gpus')

    args = parser.parse_args()

    # settings
    assert args.win_size <= args.max_seq_len, f'Window length {args.win_size} is greater than {args.max_seq_len}'
    if args.use_multi_gpu:
        assert args.use_gpu 
    assert args.encode_decode in [0,1], f'encode_decode not in {[0,1]}'
    encode = args.encode_decode == 0   # Convert to Bool
    decode = args.encode_decode == 1   # Convert to Bool
    assert args.pred_len <= args.win_size
    assert args.win_size % args.pred_len == 0

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
    else:
        args.local_rank = 0


    os.makedirs("../Results1/" +args.text_file[11:]+ f'/'+args.model_name+ f'_'+str(args.win_size)+ f'_'+str(args.pred_len)+ f'_'+str(args.batch_size),exist_ok=True)
    compressed_file_name = "../Results1/" +args.text_file[11:]+ f'/'+args.model_name+ f'_'+str(args.win_size) + f'_'+str(args.pred_len)+ f'_'+str(args.batch_size)+f'/' 

    # assert args.text_file == 0
    # load data
    # with open(args.text_file,'r',encoding = 'latin-1') as f_in:
    with open(args.text_file,'r') as f_in:
        text_input = f_in.read()

    # load model
    if args.model_name == 'Llama':
        mypredictor = Llama_process_predictor(args)
    elif args.model_name == 'Llama13-hf':
        mypredictor = Llama2_13b_process_predictor(args)
    elif args.model_name == 'Chatglm':
        mypredictor = Chatglm_process_predictor(args)
    elif args.model_name == 'gpt2':
        mypredictor = GPT2_process_predictor(args)
    elif args.model_name == 'gpt2-medium':
        mypredictor = GPT2_medium_process_predictor(args)
    elif args.model_name == 'gpt2-large':
        mypredictor = GPT2_large_process_predictor(args)
    elif args.model_name == 'xlnet-base':
        mypredictor = XLNet_base_process_predictor(args)
    elif args.model_name == 'xlnet-large':
        mypredictor = XLNet_large_process_predictor(args)
    else:
        print("Error!")


    # encode & decode
    start_time_encode = time.time() 
    if encode:   
  
        encoder = Encoder(mypredictor)
        encoder.encode(text_input, args.win_size, args.pred_len, compressed_file_name, args.batch_size)
    
        # calculate seconds
        encode_time = time.time() - start_time_encode
        print(f"Encoding is completed in {encode_time:.2f} seconds")
        with open(compressed_file_name + 'metrics.json', 'r') as f:
            params = json.load(f)
        params['Encoding time'] = encode_time
        with open(compressed_file_name + 'metrics.json','w') as f:
            json.dump(params, f, indent=4)        


    start_time_decode = time.time() 
    if decode:    

        decoder = Decoder(mypredictor)
        decoder.decode(compressed_file_name)

        # calculate seconds
        decode_time = time.time() - start_time_decode
        print(f"Decoding is completed in {decode_time:.2f} seconds")
        with open(compressed_file_name + 'metrics.json', 'r') as f:
            params = json.load(f)
        params['Decoding time'] = decode_time
        with open(compressed_file_name + 'metrics.json','w') as f:
            json.dump(params, f, indent=4)  

