#!/usr/bin/env bash


model_list=('gpt2' 'gpt2-medium' 'Chatglm' 'gpt2-large' 'xlnet-base' 'xlnet-large' 'Llama' 'Llama13-hf')


# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    echo "开始测试......"
    python main.py --text_file "../Dataset/Log1/Apache" --text_type 2 --model_name "$model_name" --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
done    
