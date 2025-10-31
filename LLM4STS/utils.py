from typing import Tuple
import torch
import json
import numpy as np
import struct
from encoders import BitInputStream

# convert series into window stride

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    assert (a.size - L)%S == 0
    nrows = (a.size - L) // S
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L+S), strides=(S * n, n), writeable=False)
 

# calculate file bits

def read_bitstream(bitin):
    temp_list = []
    while True:
        temp = bitin.read()
        if temp == -1:
            break
        temp_list += [temp]
    temp_arr = np.array(temp_list)
    final_ind = (np.where(temp_arr==1)[0][-1]).astype(int)
    final_arr = temp_arr[:final_ind+1]
    
    return final_arr


def cal_file_bits(file_name):
    file_in = open(file_name, 'rb')
    bitin = BitInputStream(file_in)
    compressed_bits = read_bitstream(bitin)
    file_in.close()
    return  compressed_bits.size


def cal_file_bytes(file_name):
    file_in = open(file_name, 'rb')
    file_bytes = file_in.read()
    file_in.close()
    return  len(file_bytes)


def cal_str_bytes(data_str):
    str_data = bytes(data_str,'ascii')
    return len(str_data)


def get_str_array(array):
    array_used = array.reshape(-1)
    str_out = str()
    for i in range(array_used.size):
        str_out +=str(array_used[i])+" "
    return str_out
 


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


# get rank and next token from probs

def gen_rank(probs,next_token):
    probs = torch.tensor(probs)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True) 
    rank_list = []
    if next_token.shape[0]>1:
        for i in range(next_token.shape[0]):
            rank_list += [torch.where(probs_idx[i:i+1,:] == next_token[i])[-1]]
        rank = torch.squeeze(torch.stack(rank_list))
    else:
        rank = torch.where(probs_idx == next_token)[-1]
    return rank


def gen_next_token(probs,rank):
    probs,rank = torch.tensor(probs),torch.tensor(rank)

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True,stable=True)
    
    next_token_list = []
    if rank.shape[0]>1:
        for i in range(rank.shape[0]):
            # 从排序后的索引中提取对应位置的 token
            token = torch.gather(
                input=probs_idx[i:i+1, :],  # 取第 i 个样本的排序索引
                dim=-1,
                index=rank[i:i+1].unsqueeze(-1)  # 取第 i 个 rank，并增加维度
            )
            next_token_list += [token]
        next_token = torch.squeeze(torch.stack(next_token_list))
    else:
        next_token = torch.gather(
            input=probs_idx,
            dim=-1,
            index=rank.unsqueeze(-1)  # 增加维度匹配 gather 的输入要求
        )
        next_token = torch.squeeze(next_token)  # 压缩冗余维度
    return next_token



# numpy <=> txt

def load_np_to_txt(mylist,file_path):
    with open(file_path, 'w') as f:
        for arr in mylist:
            f.write(str(arr))
            f.write('\n')
    f.close()

def read_np_from_txt(file_path):
    with open(file_path, 'r') as f:
        mylist = []
        for line in f:
            arr = line.strip()
            mylist.append(int(arr))
    return np.array(mylist)



def post_process_scores(generated_output):
    processed_probs = []
    
    # 遍历每个生成步骤的原始 logits
    for step_scores in generated_output:
        # 将 logits 转换为概率
        step_probs = torch.softmax(step_scores, dim=-1).detach().cpu().numpy()
        processed_probs.append(step_probs)
    
    return np.array(processed_probs)


def post_process_good_scores(generated_output, good_tokens):
    processed_probs = []
    
    # 遍历每个生成步骤的原始 logits
    for step_scores in generated_output:
        # 将 logits 转换为概率
        step_probs = torch.softmax(step_scores, dim=-1).detach().cpu()
        
        # 创建概率归零掩码
        prob_mask = torch.ones_like(step_probs)
        prob_mask[:, good_tokens] = 0  # 标记需要保留的位置
        
        # 执行归零操作
        zeroed_probs = step_probs * (1 - prob_mask)
        
        # 重新归一化（仅保留 good_tokens 的概率）
        renormalized_probs = zeroed_probs / zeroed_probs.sum(dim=-1, keepdims=True)
        
        processed_probs.append(renormalized_probs.numpy())
    
    return np.array(processed_probs)



def post_process_good_probs(step_probs, good_tokens):

    # 创建概率归零掩码
    prob_mask = np.ones_like(step_probs)
    prob_mask[good_tokens] = 0  # 标记需要保留的位置
    
    # 执行归零操作
    zeroed_probs = step_probs * (1 - prob_mask)
    
    # 重新归一化（仅保留 good_tokens 的概率）
    renormalized_probs = zeroed_probs / zeroed_probs.sum()

    return renormalized_probs 



if __name__  == "__main__":
    len_series = 30
    a = np.arange(len_series)
    timesteps = 8
    slide = 3
    if (len_series - timesteps) % slide == 0:
        ind = (len_series - timesteps) // slide
    else:
        ind = (len_series - timesteps) // slide + 1
    r = ind * slide - (len_series - timesteps)
    a = np.concatenate((a,np.full(r,0,dtype=int)))
    s = strided_app(a,timesteps,slide)
    x = s[:,:-slide]
    y = s[:,-slide:]
    print(x,y)
    print(r)
    # print(a[-r:])
