#!/usr/bin/env bash

echo "start"
python main.py --data_file '../Dataset/Float/gps.csv' --decimals 6
echo "start"
python main.py --data_file '../Dataset/Float/ZH_root.csv' --decimals 2
echo "start"
python main.py --data_file '../Dataset/Float/city_temp.csv' --decimals 1
