import os
import warnings
import argparse
import sys

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 从本地导入
from predictors import TimesFM_predictor
from hybrid_model_compressor import HybridEncoder, HybridDecoder

# 从TimesFM4NTS导入utils
timesfm_dir = os.path.join(parent_dir, 'TimesFM4NTS')
if timesfm_dir not in sys.path:
    sys.path.insert(0, timesfm_dir)
from utils import read_list_from_csv, load_list_to_csv

import torch
import torch.distributed as dist
import numpy as np
import time
import json

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # ==================== 可修改的文件名变量 ====================
    # 修改这里的文件名来压缩不同的CSV文件
    CSV_FILENAME = "gps.csv"  # 修改为你想要压缩的文件名
    CSV_DATA_DIR = "../csv_data"  # CSV数据目录路径
    # ============================================================

    parser = argparse.ArgumentParser(description='Hybrid Model Compressor for NTS')
    
    # base
    parser.add_argument('--model_name', type=str, default='TimesFM', help='the name of model')
    parser.add_argument('--data_file', type=str, default=None, help='data file path (will use CSV_FILENAME if not specified)')
    parser.add_argument('--data_flag', type=int, default=0, help='1:int,0:float')
    parser.add_argument('--decimals', type=int, default=6, help='0:int,1:city_temp,2:zh_root,6:gps')
    parser.add_argument('--win_size', type=int, default=256, help='the context window length')
    parser.add_argument('--pred_len', type=int, default=1, help='the step size of sliding')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--encode_decode', type=int, default=0, help='0: encode, 1: decode')

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
        raise IOError(f"Unsupported model name: {args.model_name}")
    
    if args.use_multi_gpu:
        assert args.use_gpu 
    assert args.encode_decode in [0,1], f'encode_decode not in {[0,1]}'
    encode = args.encode_decode == 0   # Convert to Bool
    decode = args.encode_decode == 1   # Convert to Bool

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        args.local_rank = local_rank
    else:
        args.local_rank = 0

    # 确定数据文件路径
    if args.data_file is None:
        # 使用变量中指定的文件名
        csv_data_path = os.path.join(parent_dir, CSV_DATA_DIR.lstrip('../'))
        if not os.path.exists(csv_data_path):
            # 尝试其他可能的路径
            csv_data_path = os.path.join(current_dir, CSV_DATA_DIR.lstrip('../'))
        if not os.path.exists(csv_data_path):
            csv_data_path = CSV_DATA_DIR
        
        args.data_file = os.path.join(csv_data_path, CSV_FILENAME)
        if not os.path.exists(args.data_file):
            print(f"警告: 数据文件不存在: {args.data_file}")
            print(f"请检查 CSV_FILENAME 和 CSV_DATA_DIR 变量，或使用 --data_file 参数指定完整路径")
    
    print(f"使用数据文件: {args.data_file}")
    
    # 创建结果目录
    data_file_name = os.path.basename(args.data_file)
    data_file_base = os.path.splitext(data_file_name)[0]  # 去掉扩展名
    
    results_dir = os.path.join(current_dir, 'results')
    output_subdir = os.path.join(results_dir, data_file_base, 
                                 f'{args.model_name}_{args.win_size}_{args.pred_len}_{args.batch_size}')
    os.makedirs(output_subdir, exist_ok=True)
    compressed_file_name = os.path.join(output_subdir, '')  # 添加末尾斜杠
    
    print(f"结果将保存到: {compressed_file_name}")

    # load data
    print(f"正在加载数据: {args.data_file}")
    data_series = read_list_from_csv(args.data_file, args.data_flag)
    print(f"数据加载完成，共 {len(data_series)} 个数据点")
    print(f"前10个数据点: {data_series[:10]}")

    # load model
    print("正在加载 TimesFM 模型...")
    mypredictor = TimesFM_predictor(args)
    print("模型加载完成")

    # encode & decode
    start_time_encode = time.time() 
    if encode:   
        print("\n开始编码（压缩）...")
        encoder = HybridEncoder(mypredictor)
        encoder.encode(data_series, args.win_size, args.pred_len, compressed_file_name, args.batch_size)
        
        # calculate seconds
        encode_time = time.time() - start_time_encode
        print(f"编码完成，耗时: {encode_time:.2f} 秒")
        
        # 保存编码时间到参数文件
        params_name = compressed_file_name + "params"
        if os.path.exists(params_name):
            with open(params_name, 'r') as f:
                params = json.load(f)
            params['Encoding time'] = encode_time
            with open(params_name, 'w') as f:
                json.dump(params, f, indent=4)
        else:
            # 如果参数文件不存在，创建一个
            params = {'Encoding time': encode_time}
            with open(params_name, 'w') as f:
                json.dump(params, f, indent=4)

    start_time_decode = time.time() 
    if decode:    
        print("\n开始解码（解压缩）...")
        decoder = HybridDecoder(mypredictor)
        decoded_data = decoder.decode(compressed_file_name)
        
        # 验证解码结果
        decoder.verify_text(data_series, decoded_data)

        # calculate seconds
        decode_time = time.time() - start_time_decode
        print(f"解码完成，耗时: {decode_time:.2f} 秒")
        
        # 保存解码时间到参数文件
        params_name = compressed_file_name + "params"
        if os.path.exists(params_name):
            with open(params_name, 'r') as f:
                params = json.load(f)
            params['Decoding time'] = decode_time
            with open(params_name, 'w') as f:
                json.dump(params, f, indent=4)
        
        # 保存解码结果
        decoded_output_file = os.path.join(output_subdir, f'{data_file_base}_decoded.csv')
        load_list_to_csv(decoded_data, decoded_output_file, args.data_flag, args.decimals)
        print(f"解码结果已保存到: {decoded_output_file}")
