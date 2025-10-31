#!/usr/bin/env bash


model_list=(
            # 'Llama'
            # 'Llama13-hf' \
            # 'gpt2'
            'gpt2-medium'
            # 'Chatglm' \
            # 'gpt2-large' \
            # 'xlnet-base' \
            # 'xlnet-large'
            )
# pred_len_list=(32 16 8 4 2 1)
pred_len_list=(32 16 8 4 2)

# 遍历数据集列表
for model_name in "${model_list[@]}"
do
    echo "开始测试......"
    for pred_len in "${pred_len_list[@]}"
    do
    python main.py --text_file "../Dataset/TH-Climate/text_syndata_climate0" --model_name "$model_name" --gpu 0 --win_size 256 --pred_len $pred_len --batch_size 8
    done
    echo "结束测试......"
done    
