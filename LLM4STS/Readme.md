### 1. 代码说明
基于LLM的整型/浮点型数据编码压缩（使用算术编码）

### 2. 使用方式
conda activate transformers

1) 下载llama模型到自定义文件夹
2) 运行代码
1、if use one gpu:
python main.py
2、if use multiple gpus:
export CUDA_VISIBLE_DEVICES=0,2,3   # 设置可用gpu
torchrun --nproc_per_node k main.py

3) 后台运行
nohup bash run30.sh > test30.log 2>&1 &
nohup bash run31.sh > test31.log 2>&1 &
nohup bash run32.sh > test32.log 2>&1 &
nohup bash run33.sh > test33.log 2>&1 &
[1] 3388438
ps aux |grep python

nohup bash run21.sh > test.log 2>&1 &