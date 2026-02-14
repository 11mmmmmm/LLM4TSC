"""
从 HuggingFace 下载 TimesFM 模型到本地 models 目录
"""
import os
from pathlib import Path

def download_timesfm_model(model_name="google/timesfm-1.0-200m-pytorch", local_dir=None):
    """
    从 HuggingFace 下载 TimesFM 模型（已废弃，请使用 download_with_huggingface_hub）
    
    此函数已废弃，请使用 download_with_huggingface_hub 方法
    """
    print("警告: 此方法已废弃，请使用 --method huggingface_hub")
    return download_with_huggingface_hub(model_name, local_dir)


def download_with_huggingface_hub(model_name="google/timesfm-1.0-200m-pytorch", local_dir=None):
    """
    使用 huggingface_hub 直接下载模型文件到本地
    
    Args:
        model_name: HuggingFace 模型名称，默认为 google/timesfm-1.0-200m-pytorch (PyTorch版本)
        local_dir: 本地保存目录，如果为 None 则使用当前脚本所在目录下的 TimesFM 子目录
    """
    try:
        from huggingface_hub import snapshot_download
        
        if local_dir is None:
            # 使用当前脚本所在目录下的 TimesFM 子目录
            script_dir = Path(__file__).parent
            local_dir = str(script_dir / "TimesFM")
        
        print(f"正在从 HuggingFace 下载模型: {model_name}")
        print(f"保存目录: {local_dir}")
        
        # 如果目录已存在，询问是否删除旧文件
        if os.path.exists(local_dir):
            print(f"警告: 目录 {local_dir} 已存在")
            print(f"将删除旧文件并重新下载...")
            import shutil
            shutil.rmtree(local_dir)
        
        # 确保目录存在
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载整个模型仓库
        print("开始下载...")
        cache_dir = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n模型下载完成！")
        print(f"模型保存在: {cache_dir}")
        print(f"\n模型文件列表:")
        if os.path.exists(local_dir):
            total_size = 0
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    total_size += size
                    rel_path = os.path.relpath(file_path, local_dir)
                    print(f"  - {rel_path} ({size:.2f} MB)")
            print(f"\n总大小: {total_size:.2f} MB")
        
        # 查找可能的模型检查点文件
        ckpt_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.ckpt') or 'checkpoint' in file.lower() or 'torch_model' in file.lower():
                    ckpt_files.append(os.path.join(root, file))
        
        if ckpt_files:
            print(f"\n找到检查点文件:")
            for ckpt in ckpt_files:
                rel_path = os.path.relpath(ckpt, local_dir)
                file_size = os.path.getsize(ckpt) / (1024 * 1024)  # MB
                print(f"  - {rel_path} ({file_size:.2f} MB)")
            print(f"\n✓ 模型已下载完成！")
            print(f"TimesFM_predictor.py 会自动检测并使用这些文件")
        else:
            print(f"\n警告: 未找到 .ckpt 文件")
            print(f"请检查模型仓库 {model_name} 是否包含正确的模型文件")
        
        return cache_dir
        
    except ImportError:
        print("未安装 huggingface_hub，正在尝试安装...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        print("安装完成，重新尝试下载...")
        return download_with_huggingface_hub(model_name, local_dir)
    except Exception as e:
        print(f"\n下载过程中出现错误: {e}")
        print(f"\n提示:")
        print(f"1. 请检查网络连接")
        print(f"2. 确认模型名称是否正确: {model_name}")
        print(f"3. 如果在中国大陆，可能需要配置代理或使用镜像")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 TimesFM 模型到 models/TimesFM 目录")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/timesfm-1.0-200m-pytorch",
        help="HuggingFace 模型名称 (默认: google/timesfm-1.0-200m-pytorch，PyTorch版本)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="本地保存目录 (默认: models/TimesFM 目录)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["timesfm", "huggingface_hub"],
        default="huggingface_hub",
        help="下载方法 (默认: huggingface_hub)"
    )
    
    args = parser.parse_args()
    
    if args.method == "huggingface_hub":
        download_with_huggingface_hub(args.model_name, args.local_dir)
    else:
        download_timesfm_model(args.model_name, args.local_dir)
