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

checkpoint_path = "model_weight.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model'])

model.eval()
model.to(device)

########################################################################

num_return_sequences = 10
max_length = 30

torch.manual_seed(42)
torch.cuda.manual_seed(42)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)                        # (B, T, vocab_size)
        logits = logits[:, -1, :]                   # 取最新生成 (B, vocab_size)
        probs = F.softmax(logits, dim = -1)         # 得到概率
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)  # top-k采样，保留最可能的50个概率
        ix = torch.multinomial(topk_probs, 1)       # 从top-50中取一个
        xcol = torch.gather(topk_indices, -1, ix)   # 根据 ix 从 topk_indices 中取出对应的 token 索引
        x = torch.cat((x, xcol), dim = 1)           # 将生成的 token 追加到序列中

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist() # 得到第i句的所有token
    decode = enc.decode(tokens)         # 解码为单词
    print(">", decode)                  # 打印

sys.exit(0)