from .Predictor import Predictor
import timesfm
import numpy as np
import torch
import time
import os

# path = "/data/liuzhiheng"


class TimesFM_predictor(Predictor):
    def __init__(self,args): 

        # model parameters
        # 使用本地 models/TimesFM 目录中的模型
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(script_dir, 'models', 'TimesFM')
        
        # 检查 models/TimesFM 目录中是否有模型文件
        self.ckpt_path = None
        
        # 查找本地模型文件
        if os.path.exists(model_dir):
            # 查找 .ckpt 文件（PyTorch 格式）
            for root, dirs, files in os.walk(model_dir):
                # 跳过 TensorFlow checkpoint 目录和缓存目录
                if 'checkpoint_' in root or '/state/' in root or '/.cache/' in root:
                    continue
                for file in files:
                    # 只查找 .ckpt 文件或 torch_model 文件
                    if file.endswith('.ckpt') or 'torch_model' in file.lower():
                        potential_path = os.path.join(root, file)
                        # 检查文件大小，太小的文件可能是损坏的
                        if os.path.getsize(potential_path) > 10 * 1024 * 1024:  # 至少 10MB
                            self.ckpt_path = potential_path
                            break
                if self.ckpt_path:
                    break
            
            if self.ckpt_path and os.path.exists(self.ckpt_path):
                print(f"找到本地模型文件: {self.ckpt_path}")
            else:
                print(f"警告: models/TimesFM 目录存在，但未找到有效的 .ckpt 文件")
                print(f"请运行以下命令下载模型:")
                print(f"  cd {os.path.dirname(model_dir)}")
                print(f"  python3 download_timesfm.py --model_name google/timesfm-1.0-200m-pytorch")
        else:
            print(f"警告: 未找到 models/TimesFM 目录")
            print(f"请运行以下命令下载模型:")
            print(f"  cd {os.path.dirname(model_dir)}")
            print(f"  python3 download_timesfm.py --model_name google/timesfm-1.0-200m-pytorch")
        self.num_layers = 20  # timesfm-1.0-200m-pytorch 模型的正确层数
        self.use_positional_embedding = False
        self.context_len = 512  # 可以设置，但模型实际最大支持512
        self.freq = 0
        # 0: high frequency such as daily, 1: weekly and monthly, 
        # 2: low frequency, short horizon time series. 

        super().__init__(args)


    def _build_model(self):
        
        start_time = time.time()

        if self.args.use_multi_gpu:
            self.args.use_multi_gpu = False
        
        self.args.device = self.device
        # torch.cuda.set_device(self.device)

        # 检查是否有本地模型文件
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"未找到有效的模型文件。请先运行下载脚本:\n"
                f"  cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models\n"
                f"  python3 download_timesfm.py --model_name google/timesfm-1.0-200m-pytorch"
            )
        
        # 使用本地模型文件加载
        try:
            print(f"正在从本地文件加载模型: {self.ckpt_path}")
            checkpoint = timesfm.TimesFmCheckpoint(path=self.ckpt_path)
            
            model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=self.args.win_size,  # 32
                    horizon_len=self.args.pred_len,          # 128
                    num_layers= self.num_layers ,
                    use_positional_embedding= self.use_positional_embedding ,
                    context_len= self.context_len,
                ),
                checkpoint=checkpoint
            )
            
            self.model = model
            print(f"{self.device}: 模型加载成功，耗时 {time.time() - start_time:.2f} 秒")
            
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e) or "Error(s) in loading state_dict" in str(e):
                print(f"\n错误: 本地模型文件不兼容或损坏: {self.ckpt_path}")
                print(f"错误信息: {str(e)[:200]}...")
                print(f"\n解决方案:")
                print(f"1. 删除旧的模型文件: rm -rf {os.path.dirname(self.ckpt_path)}")
                print(f"2. 重新下载正确的模型:")
                print(f"   cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/models")
                print(f"   python3 download_timesfm.py --model_name google/timesfm-1.0-200m-pytorch")
                raise RuntimeError(f"模型文件不兼容，请重新下载") from e
            else:
                raise
        except Exception as e:
            print(f"\n错误: 模型加载失败: {e}")
            raise
        
        return


    def model_predict(self,input_times):
        # print(input_times)

        # transform
        freqs = torch.tensor([self.freq] * len(input_times))

        # predict
        outputs,_ =  self.model.forecast(input_times,freqs)

        # print(outputs)
        if self.args.data_flag == 1:
            outputs = np.array(outputs,dtype = 'int')
        else:
            outputs = np.array(outputs,dtype = 'float')
        # assert input_times == 0

        return outputs

    