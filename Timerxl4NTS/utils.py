import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from functools import partial
import torch
import struct
import warnings
warnings.filterwarnings("ignore")


# =============================== LTSM layers ===================================


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask



# =============================== Load data ===================================


def convert_value(value,flag = 1):
    """
    根据数据类型将字符串转换为对应的数值类型
    """
    if flag == 1:
        try:
            # 尝试转换为整数
            return int(value)
        except ValueError:
            try:
                # 尝试转换为浮点数
                return float(value)
            except ValueError:
                # 如果既不是整数也不是浮点数，则保持字符串形式
                return value
    
    else:
        return float(value)

def read_list_from_csv(filename,flag = 1):
    
    with open(filename, 'r') as f:
        all_data = []
        for line in f:
            value = line.strip().split(',')[-1]
            all_data.append(convert_value(value,flag))

    return np.array(all_data)

def read_list_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        my_list = json.load(json_file)
    
    return my_list

def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        mylist = []
        for line in f:
            arr = int(line.strip().split()[0])
            mylist.append(arr)
    return np.array(mylist)


def load_list_to_json(my_list,json_file_path):
    with open(json_file_path,'w') as json_file:
        json.dump(my_list, json_file)

def load_list_to_txt(mylist,file_path):
    with open(file_path, 'w') as f:
        for arr in mylist:
            np.savetxt(f, arr, fmt='%.0f', newline=' ')
            f.write('\n')

def load_list_to_csv(mylist,file_path,flag=1,decimals=1):
    if flag == 1:
        np.savetxt(file_path, mylist,fmt='%d', delimiter=',')
    else:
        np.savetxt(file_path, mylist,fmt='%.'+str(decimals)+'f', delimiter=',')



def plot_series(GroundTruth, pred_list, residuals_list):

    time = range(len(GroundTruth))
    
    # 创建一个包含三个子图的图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制观测值和模拟值对比图
    ax1.plot(time, GroundTruth, 'b-', label='GroundTruth', alpha=0.7)
    ax1.plot(time, pred_list, 'r-', label='Predicted', alpha=0.7)
    ax1.set_title(f'GroundTruth vs Predicted Values')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制残差时间序列图
    ax2.plot(time, residuals_list, 'g-', label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_title('Residuals Time Series')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Residuals')
    ax2.grid(True)
    ax2.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图像
    plt.show()

def plot_multi_time_series(time_series_data, init_pred_list, init_residuals_list, var_index=0):
    """
    绘制时间序列分析图 
    
    参数:
    time_series_data: 列表的列表，每个子列表包含一个变量的观测值
    pred_list: 模拟值列表
    residuals_list: 残差列表
    var_index: 要分析的变量索引,默认为0,即第一个变量
    """
    time_series_data = np.array(time_series_data).transpose(1,0)
    init_pred_list = np.array(init_pred_list).transpose(1,0)
    init_residuals_list = np.array(init_residuals_list).transpose(1,0)

    # 创建时间索引（假设时间间隔相等）
    time = range(len(time_series_data[var_index]))
    GroundTruth = time_series_data[var_index]
    pred_list = init_pred_list[var_index]
    residuals_list = init_residuals_list[var_index]

    
    # 创建一个包含三个子图的图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制观测值和模拟值对比图
    ax1.plot(time, GroundTruth, 'b-', label='GroundTruth', alpha=0.7)
    ax1.plot(time, pred_list, 'r--', label='Predicted', alpha=0.7)
    ax1.set_title(f'Variable {var_index+1}: GroundTruth vs Predicted Values')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制残差时间序列图
    ax2.plot(time, residuals_list, 'g-', label='Residuals')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.set_title('Residuals Time Series')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Residuals')
    ax2.grid(True)
    ax2.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图像
    plt.show()



# =============================== convert data series into window stride ===================================

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    assert (a.size - L)%S == 0
    nrows = (a.size - L) // S
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L+S), strides=(S * n, n), writeable=False)
 

# =============================== SerializerSettings ===================================

# Serializer

# print(vec_num2repr(np.array([1.1,2.2,3.4 ,4.0]), 10, 3, 3000))
# 小数点后：prec = 3 位； 小数点前：ceil(lg3000) = 4 位
# 返回的sign是符号位
# (array([1, 1, 1, 1]), 
# array([[0, 0, 0, 1, 1, 0, 0],
    #    [0, 0, 0, 2, 2, 0, 0],
    #    [0, 0, 0, 3, 3, 9, 9],
    #    [0, 0, 0, 4, 0, 0, 0]]))

def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.
    
    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]

    sign = 1 * (val >= 0) - 1 * (val < 0)   # 正数标记1，负数标记-1
    val = np.abs(val)                       # 取绝对值
    
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base**(max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits


def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped/base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5/base**prec

    return sign * val


# vec_repr2num(np.array([1, 1, 1, 1]), np.array([[0, 0, 0, 1, 1, 0, 0],
#        [0, 0, 0, 2, 2, 0, 0],
#        [0, 0, 0, 3, 3, 9, 9],
#        [0, 0, 0, 4, 0, 0, 0]]), 10, 3, True)


@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """
    base: int = 10          # 10进制
    prec: int = 3           # 小数点后的精度
    max_val: float = 1e7    # 最大值不能超过1e7

    fixed_length: bool = False      # 是否保证所有序列化字符串长度相同
    
    time_sep: str = ' ,'         # time steps之间的分隔符号
    bit_sep: str = ' '           # 每个数字之间的分隔符号
    decimal_point: str = ''      # 小数点表示
    missing_str: str = ' Nan'    # 缺失值表示

    signed: bool = True          # 是否允许符号位的存在
    plus_sign: str = ''          # 正号表示
    minus_sign: str = ' -'       # 负号表示
    
    half_bin_correction: bool = True    # 是否在deserial时应用半区间校正


    def get_all_attributes(self):
        tmp = {
            'base':  self.base,       
            'prec':   self.prec,      
            'max_val': self.max_val, 
            'fixed_length': self.fixed_length, 
            'time_sep':  self.time_sep,   
            'bit_sep':  self.bit_sep,     
            'decimal_point': self.decimal_point,   
            'missing_str': self.missing_str, 
            'signed': self.signed,    
            'plus_sign':  self.plus_sign,     
            'minus_sign': self.minus_sign,   
            'half_bin_correction': self.half_bin_correction,   
        }
        return tmp



def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in num2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"
    
    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign
    
    # num -> repre
    # np.array([1.1,2.2,3.4 ,4.0])  ->  array([[0, 0, 0, 1, 1, 0, 0],
                                          #    [0, 0, 0, 2, 2, 0, 0],
                                          #    [0, 0, 0, 3, 3, 9, 9],
                                          #    [0, 0, 0, 4, 0, 0, 0]])      
    vnum2repr = partial(vec_num2repr,base=settings.base,prec=settings.prec,max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr),np.zeros_like(arr),arr))
    ismissing = np.isnan(arr)
    
    def tokenize(arr):
        return ''.join([settings.bit_sep+str(b) for b in arr])
    
    bit_strs = []
    for sign, digits,missing in zip(sign_arr, digits_arr, ismissing):
        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        # add bit_sep
        digits = tokenize(digits)
        # handle plus and minus signs
        sign_sep = plus_sign if sign==1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    # add time_sep
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep # otherwise there is ambiguity in number of digits in the last time step
    return bit_str


def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise 
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num,base=settings.base,prec=settings.prec,half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val)/np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign==1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep=='':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(bits) == max_bit_pos+settings.prec, f"fixed length bit_str must have {max_bit_pos+settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b==settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            #digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i-2:i+5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0]*(max_len-len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None


# Scaler
@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x    

def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]  #过滤掉 NaN 值
    if basic:
        # 计算 history 绝对值的 alpha 分位数，并与 0.01 取最大值。
        q = np.maximum(np.quantile(np.abs(history), alpha),.01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        # 计算 history 的最小值经过 beta 缩放后的偏移量。
        min_ = np.min(history) - beta*(np.max(history)-np.min(history))
        # 计算 history 减去偏移量 min_ 后的 alpha 分位数
        q = np.quantile(history-min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)


# handle the length of prediction
def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]


# 赋值
def get_SerializerSettings(myset):

    return SerializerSettings(
                    base=myset['base'], 
                    prec=myset['prec'], 
                    max_val=myset['max_val'],
                    fixed_length=myset['fixed_length'],
                    time_sep=myset['time_sep'],
                    bit_sep= myset['bit_sep'],
                    decimal_point= myset['decimal_point'],
                    missing_str=myset['missing_str'],
                    signed=myset['signed'], 
                    plus_sign=myset['plus_sign'], 
                    minus_sign=myset['minus_sign'],
                    half_bin_correction=myset['half_bin_correction'], 
            )



# encode & decode length of every-temp-file

def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len&127
        byte_str_len >>= 7
        if byte_str_len == 0:
                f.write(struct.pack('B',this_byte))
                break
        f.write(struct.pack('B',this_byte|128))
        byte_str_len -= 1


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len



# =============================== Other utils ===================================


def get_str_array(array):
    array_used = array.reshape(-1)
    str_out = str()
    for i in range(array_used.size):
        str_out +=str(array_used[i])+" "
    return str_out




if __name__ == '__main__':

    # data = np.array([999.,999,-2,1,10])
    # serializer_settings = SerializerSettings(base=10, 
    #                             prec=0, signed=True, fixed_length=False,
    #                             max_val=1000000000.0, time_sep=',', bit_sep='',
    #                             plus_sign='', minus_sign=' -',half_bin_correction=True, 
    #                             decimal_point='', missing_str=' Nan')
    # print(serialize_arr(data,serializer_settings))
    # print(convert_value(37.5,0))
    load_list_to_csv(np.array([1.3,2.4]),"test.csv",0)

