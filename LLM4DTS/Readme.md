### 1. 代码说明
基于LLM的日志文本数据编码压缩（使用算术编码）

### 2. 使用方式
conda activate transformers
or
conda activate llmzip

1) 下载llama模型到自定义文件夹
2) 运行代码
1、if use one gpu:
python main.py
2、if use multiple gpus:
export CUDA_VISIBLE_DEVICES=0,2,3   # 设置可用gpu
torchrun --nproc_per_node k main.py

3) 后台运行
nohup bash run21.sh > test1.log 2>&1 &
nohup bash run22.sh > test2.log 2>&1 &
nohup bash run23.sh > test3.log 2>&1 &
[1] 3388438
ps aux |grep python

nohup bash run.sh > test.log 2>&1 &