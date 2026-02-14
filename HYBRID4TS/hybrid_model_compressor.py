import numpy as np
import sys
import os

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 从本地predictors导入
from predictors import Predictor, TimesFM_predictor

# 从TimesFM4NTS导入utils（如果需要）
timesfm_dir = os.path.join(parent_dir, 'TimesFM4NTS')
if timesfm_dir not in sys.path:
    sys.path.insert(0, timesfm_dir)
from utils import load_list_to_csv, read_list_from_csv

# 导入算术编码
llm4sts_dir = os.path.join(parent_dir, 'LLM4STS')
if llm4sts_dir not in sys.path:
    sys.path.insert(0, llm4sts_dir)
from encoders.arithmeticcoding import (
    ArithmeticEncoder, ArithmeticDecoder,
    BitOutputStream, BitInputStream
)

import json
import torch
from tqdm import tqdm
import csv
import struct


class HybridEncoder:
    def __init__(self, timesfm_predictor: TimesFM_predictor):
        """
        混合编码器：三级预测策略
        1. 简单预测：使用前一个点的值
        2. 线性预测：如果简单预测误差>3sigma，使用前两个点做线性外推
        3. TimesFM预测：如果线性预测误差仍>3sigma，使用TimesFM模型
        
        Args:
            timesfm_predictor: TimesFM预测器实例
        """
        self.timesfm_predictor = timesfm_predictor
        self.predictor = timesfm_predictor  # 为了兼容原有接口
        
    def encode(self, data_series, timesteps, slide, compressed_file_name: str, batch_size):
        """
        编码方法：逐个处理数据点，根据误差决定使用哪种预测方法
        
        Args:
            data_series: 输入时间序列数据
            timesteps: 时间步长（窗口大小）
            slide: 滑动步长（预测长度）
            compressed_file_name: 压缩文件保存路径
            batch_size: 批次大小（在此实现中主要用于兼容，实际逐个处理）
        """
        self.compressed_file_name = compressed_file_name
        self.timesteps = timesteps
        self.slide = slide
        self.data = data_series
        
        # 处理数据集
        len_series = len(data_series)
        if (len_series - timesteps) % slide == 0:
            ind = (len_series - timesteps) // slide
        else:
            ind = (len_series - timesteps) // slide + 1
        rem = ind * slide - (len_series - timesteps)
        
        # 填充数据
        if self.predictor.args.data_flag == 1:
            data_series = np.concatenate((data_series, np.full(rem, 0, dtype=int)))
        else:
            data_series = np.concatenate((data_series, np.full(rem, 0, dtype=float)))
        
        # 保存参数（先创建，后面会更新）
        params_name = compressed_file_name + "params"
        params = {}
        params['len_series'] = len_series
        params['bs'] = batch_size
        params['timesteps'] = timesteps
        params['slide'] = slide
        
        # 创建临时目录存储初始数据
        temp_dir = self.compressed_file_name + 'temp'
        if os.path.exists(temp_dir):
            os.system("rm -r {}".format(temp_dir))
        self.temp_file_prefix = temp_dir + "/compressed"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 第一步：计算3sigma阈值
        print("Computing 3sigma threshold...")
        sigma = self._compute_sigma(data_series, timesteps, slide)
        threshold = 3 * sigma
        print(f"Sigma: {sigma:.6f}, 3sigma threshold: {threshold:.6f}")
        
        # 保存阈值到参数文件
        params['sigma'] = float(sigma)
        params['threshold'] = float(threshold)
        with open(params_name, 'w') as f:
            json.dump(params, f, indent=4)
        
        # 第二步：正式编码
        print("Compressing with hybrid model...")
        residuals_array, methods_array = self._encode_hybrid(data_series, timesteps, slide, threshold)
        print("Done")
        
        # 保存参数（包含数据长度）
        num_values = len(methods_array.reshape(-1))
        params['num_values'] = num_values
        params['sigma'] = float(sigma)
        params['threshold'] = float(threshold)
        with open(params_name, 'w') as f:
            json.dump(params, f, indent=4)
        
        # 保存为CSV（两列：预测方法类型，残差值）
        print("Saving to CSV...")
        self._save_to_csv(methods_array, residuals_array, compressed_file_name)
        
        # 压缩数据
        print("Compressing data...")
        self._compress_data(methods_array, residuals_array, compressed_file_name, num_values)
        
        # 计算压缩比
        self.compute_compression_ratio()
        
        return
    
    def _compute_sigma(self, data_series, timesteps, slide):
        """
        计算残差的标准差（sigma）
        使用简单预测（前一个点）计算所有残差，然后计算标准差
        """
        residuals = []
        
        # 使用滑动窗口处理数据
        num_windows = (len(data_series) - timesteps) // slide + 1
        
        for i in range(num_windows):
            start_idx = i * slide
            end_idx = start_idx + timesteps + slide
            
            if end_idx > len(data_series):
                break
            
            # 获取输入窗口和目标窗口
            input_window = data_series[start_idx:start_idx + timesteps]
            target_window = data_series[start_idx + timesteps:end_idx]
            
            # 简单预测：使用前一个点的值（即输入窗口的最后一个值）
            simple_pred = np.full(slide, input_window[-1])
            
            # 计算残差
            residual = target_window - simple_pred
            residuals.extend(residual)
        
        # 计算标准差
        if len(residuals) > 0:
            sigma = np.std(residuals)
        else:
            sigma = 1.0  # 默认值
        
        return sigma
    
    def _encode_hybrid(self, data_series, timesteps, slide, threshold):
        """
        混合编码：逐个点预测，根据误差决定使用哪种预测方法
        
        Args:
            data_series: 输入时间序列
            timesteps: 时间步长
            slide: 滑动步长
            threshold: 3sigma阈值
        """
        num_windows = (len(data_series) - timesteps) // slide + 1
        
        # 初始化残差列表和预测方法类型列表
        if self.predictor.args.data_flag == 1:
            residuals_list = []
        else:
            residuals_list = []
        prediction_methods_list = []  # 0=简单预测, 1=线性预测, 2=TimesFM
        
        # 保存第一个窗口的初始数据
        first_window = data_series[:timesteps]
        np.save(self.temp_file_prefix + "True0.npy", first_window)
        
        # 逐个窗口处理
        for i in tqdm(range(num_windows)):
            start_idx = i * slide
            end_idx = start_idx + timesteps + slide
            
            if end_idx > len(data_series):
                break
            
            # 获取输入窗口和目标窗口
            input_window = data_series[start_idx:start_idx + timesteps]
            target_window = data_series[start_idx + timesteps:end_idx]
            
            # 逐个点预测
            window_residuals = []
            window_methods = []
            current_window = input_window.copy()
            
            for point_idx in range(slide):
                target_point = target_window[point_idx]
                
                # 方法1：简单预测（前一个点的值）
                simple_pred = current_window[-1]
                simple_error = np.abs(target_point - simple_pred)
                
                # 方法2：线性预测（前两个点做线性外推）
                linear_pred = None
                linear_error = float('inf')
                if len(current_window) >= 2:
                    slope = current_window[-1] - current_window[-2]
                    linear_pred = 2 * current_window[-1] - current_window[-2]
                    linear_error = np.abs(target_point - linear_pred)
                
                # 方法3：TimesFM预测（只预测一个点）
                timesfm_pred = None
                timesfm_error = float('inf')
                if len(current_window) >= timesteps:
                    if self.predictor.args.data_flag == 1:
                        bx = torch.from_numpy(current_window[-timesteps:].reshape(1, -1)).long()
                    else:
                        bx = torch.from_numpy(current_window[-timesteps:].reshape(1, -1)).float()
                    
                    timesfm_pred_full = self.timesfm_predictor.model_predict(bx)
                    
                    # TimesFM只预测一个点
                    if timesfm_pred_full.ndim > 1:
                        timesfm_pred_full = timesfm_pred_full.flatten()
                    
                    if len(timesfm_pred_full) > 0:
                        timesfm_pred = timesfm_pred_full[0]
                        timesfm_error = np.abs(target_point - timesfm_pred)
                
                # 选择预测方法
                # 如果都>3sigma，选择误差最小的
                errors = [simple_error, linear_error, timesfm_error]
                preds = [simple_pred, linear_pred, timesfm_pred]
                methods = [0, 1, 2]  # 对应的方法编号
                
                # 找到误差最小的预测
                valid_data = [(e, p, m) for e, p, m in zip(errors, preds, methods) if p is not None]
                if not valid_data:
                    # 如果没有有效预测，使用简单预测
                    best_pred = simple_pred
                    best_method = 0
                else:
                    # 如果都>3sigma，选择误差最小的
                    if all(e > threshold for e, _, _ in valid_data):
                        best_data = min(valid_data, key=lambda x: x[0])
                        best_pred = best_data[1]
                        best_method = best_data[2]
                    else:
                        # 否则选择第一个误差<=threshold的
                        best_data = next((d for d in valid_data if d[0] <= threshold), valid_data[0])
                        best_pred = best_data[1]
                        best_method = best_data[2]
                
                # 计算残差
                if self.predictor.args.data_flag == 1:
                    residual = target_point - best_pred
                else:
                    residual = np.around(target_point - best_pred, decimals=self.predictor.args.decimals)
                
                window_residuals.append(residual)
                window_methods.append(best_method)
                
                # 更新当前窗口（添加预测值+残差，即真实值）
                current_window = np.append(current_window, target_point)
            
            residuals_list.append(window_residuals)
            prediction_methods_list.append(window_methods)
        
        # 转换为numpy数组
        if len(residuals_list) > 0:
            residuals_array = np.array(residuals_list)
            methods_array = np.array(prediction_methods_list)
        else:
            if self.predictor.args.data_flag == 1:
                residuals_array = np.zeros((0, slide), dtype=int)
            else:
                residuals_array = np.zeros((0, slide), dtype=float)
            methods_array = np.zeros((0, slide), dtype=int)
        
        return residuals_array, methods_array
    
    def _save_to_csv(self, methods_array, residuals_array, compressed_file_name):
        """保存预测方法类型和残差值到CSV"""
        csv_file = compressed_file_name + 'data.csv'
        methods_flat = methods_array.reshape(-1)
        residuals_flat = residuals_array.reshape(-1)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'residual'])  # 表头
            for method, residual in zip(methods_flat, residuals_flat):
                writer.writerow([int(method), residual])
    
    def _compress_data(self, methods_array, residuals_array, compressed_file_name, num_values):
        """压缩数据：算术编码压缩预测方法类型，Gorilla压缩残差值"""
        methods_flat = methods_array.reshape(-1).astype(int)
        residuals_flat = residuals_array.reshape(-1)
        
        # 1. 算术编码压缩预测方法类型
        print("  Compressing methods with arithmetic coding...")
        self._compress_methods_arithmetic(methods_flat, compressed_file_name, num_values)
        
        # 2. Gorilla压缩残差值
        print("  Compressing residuals with Gorilla...")
        self._compress_residuals_gorilla(residuals_flat, compressed_file_name, num_values)
    
    def _compress_methods_arithmetic(self, methods_array, compressed_file_name, num_values):
        """使用算术编码压缩预测方法类型"""
        # 计算频率
        unique, counts = np.unique(methods_array, return_counts=True)
        freqs = np.zeros(3, dtype=np.uint64)
        for u, c in zip(unique, counts):
            freqs[int(u)] = c
        
        # 确保所有符号都有频率
        freqs = freqs + 1  # 避免零频率
        cumul = np.zeros(4, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        
        # 保存频率表（用于解码）
        freq_file = compressed_file_name + 'methods_freqs.npy'
        np.save(freq_file, freqs)
        
        # 编码
        output_file = compressed_file_name + 'methods_compressed.bin'
        with open(output_file, 'wb') as f:
            bitout = BitOutputStream(f)
            encoder = ArithmeticEncoder(32, bitout)
            
            for method in methods_array[:num_values]:
                encoder.write(cumul, int(method))
            
            encoder.finish()
            bitout.close()
    
    def _compress_residuals_gorilla(self, residuals_array, compressed_file_name, num_values):
        """使用Gorilla压缩残差值"""
        output_file = compressed_file_name + 'residuals_compressed.bin'
        
        if self.predictor.args.data_flag == 1:
            # 整数：使用变长编码
            self._compress_integers_gorilla(residuals_array[:num_values].astype(np.int64), output_file)
        else:
            # 浮点数：使用Gorilla浮点压缩
            self._compress_floats_gorilla(residuals_array[:num_values].astype(np.float64), output_file)
    
    def _compress_integers_gorilla(self, values, output_file):
        """Gorilla压缩整数（变长编码）"""
        with open(output_file, 'wb') as f:
            # 写入第一个值（未压缩）
            f.write(struct.pack('q', values[0]))
            
            prev_value = values[0]
            bit_buffer = []
            
            for value in values[1:]:
                xor = prev_value ^ value
                
                if xor == 0:
                    # 相同值：写入0
                    bit_buffer.append(0)
                else:
                    # 不同值：写入1，然后写入XOR值
                    bit_buffer.append(1)
                    # 计算XOR的位数
                    leading_zeros = 64 if xor == 0 else (64 - xor.bit_length())
                    trailing_zeros = 0
                    temp = xor
                    while temp & 1 == 0 and temp != 0:
                        trailing_zeros += 1
                        temp >>= 1
                    
                    # 写入leading zeros (6 bits)
                    for i in range(5, -1, -1):
                        bit_buffer.append((leading_zeros >> i) & 1)
                    
                    # 写入significant bits (6 bits)
                    significant_bits = 64 - leading_zeros - trailing_zeros
                    for i in range(5, -1, -1):
                        bit_buffer.append((significant_bits >> i) & 1)
                    
                    # 写入XOR值
                    xor_shifted = xor >> trailing_zeros
                    for i in range(significant_bits - 1, -1, -1):
                        bit_buffer.append((xor_shifted >> i) & 1)
                
                prev_value = value
                
                # 每8位写入一次
                while len(bit_buffer) >= 8:
                    byte_val = 0
                    for i in range(8):
                        byte_val = (byte_val << 1) | bit_buffer.pop(0)
                    f.write(struct.pack('B', byte_val))
            
            # 写入剩余位
            if bit_buffer:
                byte_val = 0
                for i, bit in enumerate(bit_buffer):
                    byte_val = (byte_val << 1) | bit
                byte_val <<= (8 - len(bit_buffer))
                f.write(struct.pack('B', byte_val))
    
    def _compress_floats_gorilla(self, values, output_file):
        """Gorilla压缩浮点数"""
        with open(output_file, 'wb') as f:
            # 写入第一个值（未压缩）
            f.write(struct.pack('d', values[0]))
            
            prev_bits = struct.unpack('Q', struct.pack('d', values[0]))[0]
            
            bit_buffer = []
            
            for value in values[1:]:
                value_bits = struct.unpack('Q', struct.pack('d', value))[0]
                xor = prev_bits ^ value_bits
                
                if xor == 0:
                    # 相同值：写入0
                    bit_buffer.append(0)
                else:
                    # 不同值：写入1，然后写入XOR值
                    bit_buffer.append(1)
                    # 计算leading zeros和trailing zeros
                    leading_zeros = 64
                    trailing_zeros = 0
                    
                    if xor != 0:
                        leading_zeros = 0
                        mask = 1 << 63
                        while (xor & mask) == 0 and leading_zeros < 64:
                            leading_zeros += 1
                            mask >>= 1
                        
                        trailing_zeros = 0
                        temp = xor
                        while (temp & 1) == 0 and temp != 0:
                            trailing_zeros += 1
                            temp >>= 1
                    
                    # 写入leading zeros (6 bits)
                    for i in range(5, -1, -1):
                        bit_buffer.append((leading_zeros >> i) & 1)
                    
                    # 写入significant bits (6 bits)
                    significant_bits = 64 - leading_zeros - trailing_zeros
                    for i in range(5, -1, -1):
                        bit_buffer.append((significant_bits >> i) & 1)
                    
                    # 写入XOR值
                    xor_shifted = xor >> trailing_zeros
                    for i in range(significant_bits - 1, -1, -1):
                        bit_buffer.append((xor_shifted >> i) & 1)
                
                prev_bits = value_bits
                
                # 每8位写入一次
                while len(bit_buffer) >= 8:
                    byte_val = 0
                    for i in range(8):
                        byte_val = (byte_val << 1) | bit_buffer.pop(0)
                    f.write(struct.pack('B', byte_val))
            
            # 写入剩余位
            if bit_buffer:
                byte_val = 0
                for i, bit in enumerate(bit_buffer):
                    byte_val = (byte_val << 1) | bit
                byte_val <<= (8 - len(bit_buffer))
                f.write(struct.pack('B', byte_val))
    
    def compute_compression_ratio(self):
        """计算压缩比"""
        uncompressed_size = self.data.nbytes * 8
        metrics = {}
        metrics['File_bits'] = uncompressed_size
        
        # 计算压缩后大小
        methods_size = os.path.getsize(self.compressed_file_name + 'methods_compressed.bin') * 8
        residuals_size = os.path.getsize(self.compressed_file_name + 'residuals_compressed.bin') * 8
        metrics['Compressed_bits'] = methods_size + residuals_size
        metrics['Compression_ratio'] = metrics['Compressed_bits'] / uncompressed_size if uncompressed_size > 0 else 0
        
        with open(self.compressed_file_name + 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)


class HybridDecoder:
    def __init__(self, timesfm_predictor: TimesFM_predictor):
        """
        混合解码器：对应HybridEncoder的解码逻辑
        根据编码时保存的预测方法标志，使用相同的预测方法重建数据
        
        Args:
            timesfm_predictor: TimesFM预测器实例
        """
        self.timesfm_predictor = timesfm_predictor
        self.predictor = timesfm_predictor  # 为了兼容原有接口
    
    def decode(self, compressed_file_name: str):
        """
        解码方法：根据编码时的逻辑重建数据
        
        Args:
            compressed_file_name: 压缩文件路径
        """
        self.compressed_file_name = compressed_file_name
        
        # 加载参数
        params_name = compressed_file_name + 'params'
        with open(params_name, 'r') as f:
            params = json.loads(f.read())
        
        batch_size = params['bs']
        self.timesteps = params['timesteps']
        len_series = params['len_series']
        self.slide = params['slide']
        threshold = params.get('threshold', 0)  # 获取阈值
        
        if (len_series - self.timesteps) % self.slide == 0:
            len_x = (len_series - self.timesteps) // self.slide
        else:
            len_x = (len_series - self.timesteps) // self.slide + 1
        rem = len_x * self.slide - (len_series - self.timesteps)
        
        # 解压数据
        print("Decompressing data...")
        methods_array, residuals_array = self._decompress_data(compressed_file_name)
        
        temp_dir = self.compressed_file_name + 'temp'
        self.temp_file_prefix = temp_dir + "/compressed"
        
        # 解码
        print("Reconstructing data...")
        tokens_full = self._decode_hybrid(len_series, methods_array, residuals_array, threshold, rem)
        print("Done")
        
        # 保存解码结果
        load_list_to_csv(tokens_full, compressed_file_name + 'decompress.csv',
                        self.predictor.args.data_flag, self.predictor.args.decimals)
        
        return tokens_full
    
    def _decompress_data(self, compressed_file_name):
        """解压数据：算术解码预测方法类型，Gorilla解压残差值"""
        # 1. 算术解码预测方法类型
        print("  Decompressing methods...")
        methods_array = self._decompress_methods_arithmetic(compressed_file_name)
        
        # 2. Gorilla解压残差值
        print("  Decompressing residuals...")
        residuals_array = self._decompress_residuals_gorilla(compressed_file_name)
        
        return methods_array, residuals_array
    
    def _decompress_methods_arithmetic(self, compressed_file_name):
        """使用算术解码解压预测方法类型"""
        input_file = compressed_file_name + 'methods_compressed.bin'
        
        # 从params获取数据长度
        params_name = compressed_file_name + 'params'
        with open(params_name, 'r') as f:
            params = json.loads(f.read())
        num_values = params['num_values']
        
        # 加载频率表
        freq_file = compressed_file_name + 'methods_freqs.npy'
        freqs = np.load(freq_file)
        cumul = np.zeros(4, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        
        # 解码
        methods = []
        with open(input_file, 'rb') as f:
            bitin = BitInputStream(f)
            decoder = ArithmeticDecoder(32, bitin)
            
            for _ in range(num_values):
                method = decoder.read(cumul, 3)
                methods.append(method)
            
            bitin.close()
        
        return np.array(methods, dtype=int)
    
    def _decompress_residuals_gorilla(self, compressed_file_name):
        """使用Gorilla解压残差值"""
        input_file = compressed_file_name + 'residuals_compressed.bin'
        
        # 从params获取数据长度
        params_name = compressed_file_name + 'params'
        with open(params_name, 'r') as f:
            params = json.loads(f.read())
        num_values = params['num_values']
        
        if self.predictor.args.data_flag == 1:
            return self._decompress_integers_gorilla(input_file, num_values)
        else:
            return self._decompress_floats_gorilla(input_file, num_values)
    
    def _decompress_integers_gorilla(self, input_file, num_values):
        """Gorilla解压整数"""
        values = []
        bit_buffer = []
        
        with open(input_file, 'rb') as f:
            # 读取第一个值
            first_value = struct.unpack('q', f.read(8))[0]
            values.append(first_value)
            prev_value = first_value
            
            # 读取剩余数据
            data = f.read()
            for byte in data:
                for i in range(7, -1, -1):
                    bit_buffer.append((byte >> i) & 1)
            
            # 解码
            idx = 0
            while len(values) < num_values and idx < len(bit_buffer):
                if idx >= len(bit_buffer):
                    break
                
                control_bit = bit_buffer[idx]
                idx += 1
                
                if control_bit == 0:
                    # 相同值
                    values.append(prev_value)
                else:
                    # 读取leading zeros
                    if idx + 6 > len(bit_buffer):
                        break
                    leading_zeros = 0
                    for i in range(6):
                        leading_zeros = (leading_zeros << 1) | bit_buffer[idx]
                        idx += 1
                    
                    # 读取significant bits
                    if idx + 6 > len(bit_buffer):
                        break
                    significant_bits = 0
                    for i in range(6):
                        significant_bits = (significant_bits << 1) | bit_buffer[idx]
                        idx += 1
                    
                    if significant_bits == 0:
                        break
                    
                    # 读取XOR值
                    if idx + significant_bits > len(bit_buffer):
                        break
                    xor_shifted = 0
                    for i in range(significant_bits):
                        xor_shifted = (xor_shifted << 1) | bit_buffer[idx]
                        idx += 1
                    
                    # 计算trailing zeros
                    trailing_zeros = 64 - leading_zeros - significant_bits
                    xor = xor_shifted << trailing_zeros
                    value = prev_value ^ xor
                    values.append(value)
                    prev_value = value
        
        return np.array(values[:num_values], dtype=np.int64)
    
    def _decompress_floats_gorilla(self, input_file, num_values):
        """Gorilla解压浮点数"""
        values = []
        bit_buffer = []
        
        with open(input_file, 'rb') as f:
            # 读取第一个值
            first_value = struct.unpack('d', f.read(8))[0]
            values.append(first_value)
            prev_bits = struct.unpack('Q', struct.pack('d', first_value))[0]
            
            # 读取剩余数据
            data = f.read()
            for byte in data:
                for i in range(7, -1, -1):
                    bit_buffer.append((byte >> i) & 1)
            
            # 解码
            idx = 0
            while len(values) < num_values and idx < len(bit_buffer):
                if idx >= len(bit_buffer):
                    break
                
                control_bit = bit_buffer[idx]
                idx += 1
                
                if control_bit == 0:
                    # 相同值
                    values.append(values[-1])
                else:
                    # 读取leading zeros
                    if idx + 6 > len(bit_buffer):
                        break
                    leading_zeros = 0
                    for i in range(6):
                        leading_zeros = (leading_zeros << 1) | bit_buffer[idx]
                        idx += 1
                    
                    # 读取significant bits
                    if idx + 6 > len(bit_buffer):
                        break
                    significant_bits = 0
                    for i in range(6):
                        significant_bits = (significant_bits << 1) | bit_buffer[idx]
                        idx += 1
                    
                    if significant_bits == 0:
                        break
                    
                    # 读取XOR值
                    if idx + significant_bits > len(bit_buffer):
                        break
                    xor_shifted = 0
                    for i in range(significant_bits):
                        xor_shifted = (xor_shifted << 1) | bit_buffer[idx]
                        idx += 1
                    
                    # 计算trailing zeros
                    trailing_zeros = 64 - leading_zeros - significant_bits
                    xor = xor_shifted << trailing_zeros
                    
                    value_bits = prev_bits ^ xor
                    value = struct.unpack('d', struct.pack('Q', value_bits))[0]
                    values.append(value)
                    prev_bits = value_bits
        
        return np.array(values[:num_values], dtype=np.float64)
    
    def _decode_hybrid(self, len_series, methods_array, residuals_array, threshold, rem):
        """
        混合解码：使用解压后的预测方法类型和残差值重建数据
        
        Args:
            len_series: 原始序列长度
            methods_array: 预测方法类型数组（已解压）
            residuals_array: 残差数组（已解压）
            threshold: 3sigma阈值
            rem: 填充长度
        """
        # 初始化重建序列
        if self.predictor.args.data_flag == 1:
            tokens_full = np.zeros(len_series + rem, dtype=int)
        else:
            tokens_full = np.zeros(len_series + rem, dtype=float)
        
        # 加载第一个窗口的初始数据
        first_window = np.load(self.temp_file_prefix + "True0.npy")
        tokens_full[:self.timesteps] = first_window
        
        # 计算窗口数量
        if (len_series - self.timesteps) % self.slide == 0:
            num_windows = (len_series - self.timesteps) // self.slide
        else:
            num_windows = (len_series - self.timesteps) // self.slide + 1
        
        # 重塑数组
        methods_2d = methods_array.reshape(num_windows, self.slide)
        residuals_2d = residuals_array.reshape(num_windows, self.slide)
        
        # 逐个窗口解码
        for i in tqdm(range(num_windows)):
            start_idx = i * self.slide
            end_idx = start_idx + self.timesteps + self.slide
            
            if end_idx > len(tokens_full):
                break
            
            # 获取当前输入窗口
            current_window = tokens_full[start_idx:start_idx + self.timesteps].copy()
            window_methods = methods_2d[i]
            window_residuals = residuals_2d[i]
            
            # 逐个点重建
            for point_idx in range(self.slide):
                method = window_methods[point_idx]
                residual = window_residuals[point_idx]
                
                # 根据预测方法类型选择预测
                if method == 0:
                    # 简单预测
                    pred = current_window[-1]
                elif method == 1:
                    # 线性预测
                    if len(current_window) >= 2:
                        slope = current_window[-1] - current_window[-2]
                        pred = 2 * current_window[-1] - current_window[-2]
                    else:
                        pred = current_window[-1]
                else:  # method == 2
                    # TimesFM预测
                    if len(current_window) >= self.timesteps:
                        if self.predictor.args.data_flag == 1:
                            bx = torch.from_numpy(current_window[-self.timesteps:].reshape(1, -1)).long()
                        else:
                            bx = torch.from_numpy(current_window[-self.timesteps:].reshape(1, -1)).float()
                        
                        timesfm_pred_full = self.timesfm_predictor.model_predict(bx)
                        
                        if timesfm_pred_full.ndim > 1:
                            timesfm_pred_full = timesfm_pred_full.flatten()
                        
                        if len(timesfm_pred_full) > 0:
                            pred = timesfm_pred_full[0]
                        else:
                            pred = current_window[-1]
                    else:
                        pred = current_window[-1]
                
                # 重建数据：预测值 + 残差
                if self.predictor.args.data_flag == 1:
                    reconstructed_point = pred + residual
                else:
                    reconstructed_point = np.around(
                        pred + residual, decimals=self.predictor.args.decimals
                    )
                
                # 更新当前窗口
                current_window = np.append(current_window, reconstructed_point)
                
                # 更新tokens_full
                tokens_full[start_idx + self.timesteps + point_idx] = reconstructed_point
        
        # 移除填充
        if rem > 0:
            tokens_full = tokens_full[:-rem]
        
        return tokens_full
    
    def verify_text(self, data_series, decoded_data):
        """验证解码结果"""
        if np.array_equal(data_series, decoded_data):
            print(f'Successful decoding')
        else:
            print("********!!!!! Error !!!!!*********")
            # 计算差异
            diff = np.abs(data_series - decoded_data)
            print(f"Max difference: {np.max(diff)}")
            print(f"Mean difference: {np.mean(diff)}")
