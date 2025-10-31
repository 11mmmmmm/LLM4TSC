#!/usr/bin/env bash

win_size_list=(8 16 32 64 128)
model_list=(
            'Llama' \
            # 'Llama13-hf' \
            # 'gpt2' \
            # 'gpt2-medium' \
            # 'Chatglm' \
            # 'gpt2-large' \
            # 'xlnet-base' \
            # 'xlnet-large'
            )


# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    echo "开始测试......"
    for win_size in "${win_size_list[@]}"
    do
    python main.py --text_file "../Dataset/Float/text_city_temp" --model_name "$model_name" --gpu 0 --win_size $win_size --pred_len 1 --batch_size 1
    done
    echo "结束测试......"
done    
