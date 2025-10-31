#!/usr/bin/env bash


# model_list=('gpt2' 'gpt2-medium')
model_list=('gpt2' 'gpt2-medium' 'gpt2-large')


# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    python main.py --text_file "../Dataset/Float/text_city_temp" --model_name "$model_name" --gpu 2 --win_size 256
    # python main.py --text_file "../Dataset/Log2/text_ZYSD_2" --model_name "$model_name" --gpu 1 --win_size 128
    # python main.py --text_file "../Dataset/Log2/text_ZYSD_3" --model_name "$model_name" --gpu 1 --win_size 86
    # python main.py --text_file "../Dataset/Log2/text_ZYSD_4" --model_name "$model_name" --gpu 1 --win_size 64
done    

