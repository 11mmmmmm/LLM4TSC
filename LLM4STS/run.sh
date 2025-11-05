#!/usr/bin/env bash

dataset_list=(
    # '../Dataset/TY-Fuel/text_syndata_fuel1' \
    # '../Dataset/Nifty-Stocks/text_syndata_stocks0' \
    # '../Dataset/Cyber-Vehicle/text_syndata_vehicle0' \
    # '../Dataset/GW-Magnetic/text_syndata_magnetic2' \
    # '../Dataset/Metro-Traffic/text_syndata_metro' \
    # '../Dataset/TY-Fuel/text_syndata_fuel0' \
    # '../Dataset/TH-Climate/text_syndata_climate0' \
    # '../Dataset/ATimeSeries-Dataset/text_IoT5'\
    # '../Dataset/USGS-Earthquakes/text_syndata_earthquakes0' \
    # '../Dataset/Cyber-Vehicle/text_syndata_vehicle25' \
)



# 遍历数据集列表
for data_file in "${dataset_list[@]}"
do
    # 执行 Python 脚本
    python main.py --model_name 'Llama13-hf' --text_file "$data_file" --text_type 2 --gpu 1 --win_size 256 --pred_len 1 --batch_size 1
    echo "llama"
    python main.py --model_name 'Llama' --text_file "$data_file" --text_type 2 --gpu 1 --win_size 256 --pred_len 1 --batch_size 1
done    

