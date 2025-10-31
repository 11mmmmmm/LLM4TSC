#!/usr/bin/env bash
model_list=('TimesFM')
pred_list=(2 4 8 16 32)


# 遍历数据集列表
for pred_len in "${pred_list[@]}"
do
    for model_name in "${model_list[@]}"
    do 
        python main.py --data_file "../Dataset/Cyber-Vehicle/syndata_vehicle25.csv" --model_name "$model_name" --gpu 1 --win_size 256 --pred_len $pred_len --batch_size 1
    done
done    

