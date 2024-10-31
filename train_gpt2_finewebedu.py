"""
优化手段：
torch.compile (在windows环境中不能使用)
flashattention 显著加快
将vocab_size增长到2幂的整数倍
确保参数的globao norm不超过1, 避免训练迭代中产生过大的参数变化
带预热的余弦衰减learning rate
使用weight decay: 强制optimization使用更多权重 避免单个权重过大
使用梯度累计: 以串行方式在小GPU上模拟任意大小的batch
多GPU情况下多线程执行
"""

import os
import math
import time
import tiktoken
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import sys



########################################################################

""" 模型定义 """

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # qkv generation
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd
        
        # qkv gen
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs), n_embd = num_head * head_size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        
        # attention
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # auto-regressive mask 确保因果性
        #att = F.softmax(att, dim = -1)
        #y = att @ v # (B, nh, T, hs)
        
        # flashattention
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# a Transformer Layer
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1   = nn.LayerNorm(config.n_embd)
        self.attn   = CausalSelfAttention(config)
        self.ln_2   = nn.LayerNorm(config.n_embd)
        self.mlp    = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTconfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # number of tokens: 50000 BPE merges, 256 bytes tokens, 1 endoftext
    n_layer:    int = 12
    n_head:     int = 12
    n_embd:     int = 768

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                wte     = nn.Embedding(config.vocab_size, config.n_embd),                  # token embedding
                wpe     = nn.Embedding(config.block_size, config.n_embd),                  # position embedding
                h       = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # create n_layer * Block
                ln_f    = nn.LayerNorm(config.n_embd)                                      # final LayerNorm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)            # final classifier (n_embd dimensions -> vocab_size words, no bias)
        
        # 权重共享（token embeding和LM head使用相同权重）
        self.transformer.wte.weight = self.lm_head.weight
        
        # 初始化权重（根据GPT-2）
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5    # 根据GPT-2论文
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @classmethod
    def from_pretrained(cls, model_type):
        # 从 huggingface 加载模型参数
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # 超参数
        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd = 768),  # 124M
            'gpt2-medium':  dict(n_layer = 24, n_head = 16, n_embd = 1024), # 350M
            'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embd = 1280), # 774M
            'g;t2-xl':      dict(n_layer = 48, n_head = 25, n_embd = 1600)  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTconfig(**config_args)   # 创建一个 minGPT 模型的配置对象
        model = GPT(config)                 # 初始化 minGPT 模型
        sd = model.state_dict()             # 获取模型的参数字典
        sd_keys = sd.keys()                 # 存储参数的名称，去除以'.attn.bias'结尾的
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # 加载 Hugging Face 预训练的 GPT2 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # 过滤 Hugging Face 模型的参数
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # 待转置的参数名称（与conv1D有关）
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 检查 minGPT 和 Hugging Face 模型的参数数量是否一致
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # 参数复制与特殊处理
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 特殊处理：转置transpoed列表中的参数
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # 此操作不计算梯度
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 获取所有参数，并过滤出期中需要更新梯度的部分
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 所有二维及以上的参数（如Matmul weight、embedding weight）应用权重衰减
        # bias, layernorm参数不衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 统计并打印参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 优先使用kernel_fused版本的AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer
    
    def forward(self, idx, targets=None):
        # idx (B, T)
        # 序列长度 T 不超过模型的最大序列长度
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # 词嵌入+位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 生成长度为 T 的位置索引
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        
        # 前向传播，得到输出 logits (logits经softmax变为概率)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logit (B, T, vocab_size)
        
        # 损失计算
        loss = None
        if targets is not None:
            # logits: (B, T, vocab_size) -> (B*T, vocab_size)
            # targets: (B, T) -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

########################################################################

""" 从预处理的edu_fineweb10B训练集加载 """
def load_tokens(filename):
    npt = np.load(filename)                         # 从文件中读取数据，存储为 Numpy 数组
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype = torch.long)     # 转换为 PyTorch 张量
    return ptt

""" 导入训练用文本, 适应于多GPU下的分布式数据加载 """

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank    # 当前进程编号
        self.num_processes = num_processes  # 进程总数
        assert split in {'train', 'val'}
        
        # 获取文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)              # 列出 data_root 目录下的所有文件
        shards = [s for s in shards if split in s]  # 过滤出包含指定 split（即 'train' 或 'val'）的文件名
        shards = sorted(shards)                     # 文件排序
        shards = [os.path.join(data_root, s) for s in shards]   # 形成完整文件路径
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        # 初始化状态参数
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)   # inputs
        y = (buf[1:]).view(B, T)    # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

########################################################################

""" 设备设置 """

ddp = int(os.environ.get('RANK', -1) != -1) # 确定是否分布式数据并行
if ddp:
    assert torch.cuda.is_available(), "need CUDA for DDP"
    init_process_group(backend = 'nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp == 0   # do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

########################################################################

""" 数据格式选择 & 初始化模型"""

# 数据格式选择
# high: TF32
torch.set_float32_matmul_precision('high')

# 初始化模型
# model = GPT(GPTconfig())
# model = GPT(GPTconfig(vocab_size = 50304))
model = GPT.from_pretrained('gpt2')
model.to(device)
# model = torch.compile(model) # 模型编译在windows上不支持！

# 多GPU条件下
if ddp:
    model = DDP(model, device_ids = {ddp_local_rank})
raw_model = model.module if ddp else model


########################################################################

""" 导入训练文本 """
""" 使用梯度累计的Batch合成方法进行训练"""

# 计算梯度累计的参数
total_batch_size = 524288 # 2^19, 约0.5M
B = 8
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# 导入文本
train_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "train")
val_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "val")

########################################################################

"""带预热的余弦衰减learning rate"""

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 10e9 / (B * T * ddp_world_size)  # 10e9 / (B * T * num_GPU)
def get_lr(it):
    # 预热阶段，线性上升到max_lr
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 后期阶段：min_lr
    if it > max_steps:
        return min_lr
    # 中间阶段，以cos函数下降到min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

########################################################################

# 加载optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8)
# optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)

# 创建日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # 清空已存在的log_file
    pass

for step in range(max_steps):
    t0 = time.time()    # 开始计时
    last_step = (step == max_steps - 1)
    
    # 每100次迭代，评估validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type = device, dtype = torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            # 记录模型权重
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # 保存模型
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)
    
    # 训练
    model.train()
    optimizer.zero_grad()
    # 梯度累计的micro-batch
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # 使用bf16数据格式
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # loss可以累计，但是loss也需要保存平方误差的均值
        loss_accum += loss.detach()
        # 多设备条件下的同步：仅在最后一个micro_step执行
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    # 多设备条件下，统计所有设备上的loss_accum的均值
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 确保参数的globao norm不超过1，避免训练迭代中产生过大的参数变化
    # 动态决定learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    # 统计
    t1 = time.time()    # 结束计时
    dt = (t1 - t0) * 1000   # 毫秒
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}" )

if ddp:
    destroy_process_group()

sys.exit(0)