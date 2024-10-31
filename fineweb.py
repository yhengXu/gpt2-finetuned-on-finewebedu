"""
下载并分片保存训练集FineWeb-Edu
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 每个片段的token大小为 100M

# 检查并创建存储数据片段的目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 下载
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2") # 使用GPT-2的tokenizer
eot = enc._special_tokens['<|endoftext|>'] # 定义结束token

# tokenize将文本转化为token数组，token值为uint16类型
def tokenize(doc):
    tokens = [eot] # 文档以eot开始
    tokens.extend(enc.encode_ordinary(doc["text"])) # 编码为token
    tokens_np = np.array(tokens)    # 转换为numpy数组
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"    # 确保token值在uint16范围内
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# 将token数组保存为numpy文件
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize文档并分片存储
if __name__ == '__main__':
    nprocs = max(1, os.cpu_count()//2)  # 线程数
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
    
            # 如果当前分片未达到100M token，则继续追加
            if token_count + len(tokens) < shard_size:
                # 追加
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # 更新进度
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            # 如果当前分片达到100M token，存储到文件并创建新的分片
            else:
                # 对于0号分片，标记为评估集；其余为训练集
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # 优先将当前分片补充道100M
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # 剩余token进入下一分片
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
    
        # 最后一个分片处理
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

