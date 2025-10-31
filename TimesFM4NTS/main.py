import os
import warnings
import argparse
from predictors import *
from compressor import Encoder,Decoder
from utils import read_list_from_csv
import torch
import torch.distributed as dist
import numpy as np
import time
import json
import random

 
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LTSM for NTS')
    
    # base
    parser.add_argument('--model_name', type=str,default='TimesFM',help = 'the name of model')
    parser.add_argument('--data_file', type=str,default='../Dataset/Float/gps.csv')
    parser.add_argument('--data_flag', type=int, default=0, help='1:int,0:float')
    parser.add_argument('--decimals', type=int, default=6, help='0:int,1:city_temp,2:zh_root,6:gps')
    parser.add_argument('--win_size', type=int, default=256, help='the context window length')
    parser.add_argument('--pred_len', type=int, default=1, help='the step size of sliding')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--encode_decode', type=int, default=0,help='0: encode, 1: decode')

    # gpu
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    # settings
    if args.model_name in ['Llama','Llama13','gpt2','gpt2-medium','gpt2-large']:
        args.model_category = 'LLM'
    elif args.model_name in ['TimesFM','Timerxl']:
        args.model_category = 'LTSM'
    else:
        raise IOError
    if args.use_multi_gpu:
        assert args.use_gpu 
    assert args.encode_decode in [0,1], f'encode_decode not in {[0,1]}'
    encode = args.encode_decode == 0   # Convert to Bool
    decode = args.encode_decode == 1   # Convert to Bool
    # assert args.pred_len <= args.win_size
    # assert args.win_size % args.pred_len == 0

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
    else:
        args.local_rank = 0

    # assert args.data_file == 0
    data_file_dir = args.data_file.split('/')[-2]
    data_file_path = args.data_file.split('/')[-1]
    os.makedirs("../Results2/"+data_file_dir+'/'+data_file_path[:-4]+ f'/{args.model_name}_{args.win_size}_{args.pred_len}_{args.batch_size}',exist_ok=True)
    compressed_file_name = "../Results2/"+data_file_dir+'/'+data_file_path[:-4]+ f'/{args.model_name}_{args.win_size}_{args.pred_len}_{args.batch_size}/' 


    # load data
    data_series = read_list_from_csv(args.data_file,args.data_flag)
    print(data_series[:10])


    # load model
    # if args.model_name == 'Llama':
    #     # mypredictor = Llama_predictor(args)
    #     pass
    # elif args.model_name == 'Llama13':
    #     # mypredictor = Llama13_predictor(args)
    #     pass
    # elif args.model_name == 'gpt2':
    #     # mypredictor = GPT2_predictor(args)
    #     pass
    # elif args.model_name == 'gpt2-medium':
    #     # mypredictor = GPT2_medium_predictor(args)
    #     pass
    # elif args.model_name == 'gpt2-large':
    #     # mypredictor = GPT2_large_predictor(args)
    #     pass
    if args.model_name == 'TimesFM':
        mypredictor = TimesFM_predictor(args)
        pass
    elif args.model_name == 'Timerxl':
        # mypredictor = Timerxl_predictor(args)
        pass


    # encode & decode
    start_time_encode = time.time() 
    if encode:   

        encoder = Encoder(mypredictor)
        encoder.encode(data_series,args.win_size,args.pred_len,compressed_file_name,args.batch_size)
        
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
        decoded_data = decoder.decode(compressed_file_name)
        decoder.verify_text(data_series,decoded_data)

        # calculate seconds
        decode_time = time.time() - start_time_decode
        print(f"Decoding is completed in {decode_time:.2f} seconds")
        with open(compressed_file_name + 'metrics.json', 'r') as f:
            params = json.load(f)
        params['Decoding time'] = decode_time
        with open(compressed_file_name + 'metrics.json','w') as f:
            json.dump(params, f, indent=4)  
        

