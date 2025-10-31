#!/usr/bin/env bash


model_list=(
            'Llama'
            # 'Llama13-hf' \
            # 'gpt2' \
            # 'gpt2-medium' \
            # 'Chatglm' \
            # 'gpt2-large' \
            # 'xlnet-base' \
            # 'xlnet-large'
            )
batch_size_list=(8 16 32 64 128)

# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    echo "开始测试......"
    for batch_size in "${batch_size_list[@]}"
    do
    python main.py --text_file "../Dataset/Float/text_city_temp" --model_name "$model_name" --gpu 0 --win_size 64 --pred_len 1 --batch_size $batch_size
    done
    echo "结束测试......"
done    
