#!/usr/bin/env bash

dataset_list=(
    '../Dataset/Float/text_city_temp'
    # '../Dataset/TY-Fuel/text_syndata_fuel1' \
    # '../Dataset/Cyber-Vehicle/text_syndata_vehicle0' \
    # '../Dataset/GW-Magnetic/text_syndata_magnetic2' \
    # '../Dataset/Metro-Traffic/text_syndata_metro' \
    # '../Dataset/TH-Climate/text_syndata_climate0' \
    # '../Dataset/USGS-Earthquakes/text_syndata_earthquakes0' \
    # '../Dataset/Nifty-Stocks/text_syndata_stocks0' \
    # '../Dataset/ATimeSeries-Dataset/text_IoT5'\
    # '../Dataset/TY-Fuel/text_syndata_fuel0' \
    # '../Dataset/Cyber-Vehicle/text_syndata_vehicle25' \
)



# 遍历数据集列表
for data_file in "${dataset_list[@]}"
do
    # 执行 Python 脚本
    python main.py --model_name 'gpt2' --text_file "$data_file" --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
    echo "end"
    python main.py --model_name 'gpt2-medium' --text_file "$data_file" --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
    echo "end"
    python main.py --model_name 'gpt2-large' --text_file "$data_file" --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
    # echo "end"
    # python main.py --model_name 'Llama' --text_file "$data_file" --gpu 3 --win_size 256 --pred_len 1 --batch_size 1
    # echo "end"
    # python main.py --model_name 'Llama13-hf' --text_file "$data_file" --gpu 3 --win_size 256 --pred_len 1 --batch_size 1

done    

