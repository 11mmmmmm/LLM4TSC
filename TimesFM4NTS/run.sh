#!/usr/bin/env bash

dataset_list=(
    '../Dataset/Nifty-Stocks/syndata_stocks0.csv' \
    '../Dataset/Cyber-Vehicle/syndata_vehicle25.csv' \
    '../Dataset/TY-Fuel/syndata_fuel0.csv' \
    '../Dataset/USGS-Earthquakes/syndata_earthquakes0.csv' \
    '../Dataset/TH-Climate/syndata_climate0.csv' \
    '../Dataset/ATimeSeries-Dataset/IoT5-hf.csv' \
)


# 遍历数据集列表
for data_file in "${dataset_list[@]}"
do
    # 执行 Python 脚本
    python main.py --data_file "$data_file" --model_name "TimesFM" --gpu 3 --win_size 256 --pred_len 1 --batch_size 1
done    

