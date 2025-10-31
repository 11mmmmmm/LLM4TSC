### 1. 代码说明
使用TimesFM和Timer执行数值时序数据编码压缩
main 入口文件
compressor -> predictor
predictor -> llama
utils 工具文件


### 2. 使用方式
(1) timerxl
conda create --name timer python=3.10
conda activate timer
pip install -r requirements.txt

# recommended parameters: 256_1_1
<!-- torchrun --nproc_per_node k main.py -->
<!-- python main.py -->
<!-- bash run.sh -->


(2) timesfm
conda create -n nts python=3.11.10
conda activate nts
source install.sh

# export CUDA_VISIBLE_DEVICES=0,1,2,3   # 设置可用gpu，否则timesfm默认使用gpu0
<!-- python main.py -->
<!-- export CUDA_VISIBLE_DEVICES=0,2,3   # 设置可用gpu -->
<!-- torchrun --nproc_per_node k main.py -->

(3) 后台运行
nohup python main.py > test128.log 2>&1 &
ps aux |grep python



