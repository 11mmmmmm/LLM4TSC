#!/usr/bin/env bash


model_list=('gpt2' 'gpt2-medium' 'gpt2-large')


# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    echo "开始测试......"
    python main.py --text_file "../Dataset/TH-Climate/text_syndata_climate0" --model_name "$model_name" --gpu 0
done    

