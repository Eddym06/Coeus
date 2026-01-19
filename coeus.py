import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# 1. Configuración Hardware-Aware (V2.5)
# =============================================================================
@dataclass
class CoeusConfig:
    block_size: int = 2048       # Contexto máximo (usado para buffer RoPE/Train)
    vocab_size: int = 50257      
    n_layer: int = 6             
    n_head: int = 6              # Cabezas de Query
    n_kv_head: int = 2           # GQA: Cabezas de Key/Value (Ratio 3:1)
    n_embd: int = 384            
    window_size: int = 128       
    dropout: float = 0.0         

# =============================================================================
# 2. Utilerías y RoPE
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm mantiene float32 internamente para precisión
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        # Ratio estándar de LLaMA: 2/3 * 4 * d_model
        hidden_dim = int(2 * 4 * config.n_embd / 3) 
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256) 

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) 
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False) 
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Computa la rotación dinámica basada en la dimensión T (seq_len)
        # x shape esperado en dim 2 es T.
        t = torch.arange(x.shape[2], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [T, D] -> Unsqueeze para [1, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Implementa Grouped Query Attention mediante expansion."""
    B, n_kv_head, T, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_head, n_rep, T, head_dim)
        .reshape(B, n_kv_head * n_rep, T, head_dim)
    )

# =============================================================================
# 3. Kernel JIT (Optimizado V2.5)
# =============================================================================
@torch.jit.script
def recurrent_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Recurrencia Lineal O(1) Memoria.
    Optimizado para evitar copias extra.
    """
    B, H, T, D = q.shape
    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    
    input_scale = 1.0 - alpha

    for t in range(T):
        # Slicing eficiente
        q_t = q[:, :, t, :].unsqueeze(2)
        k_t = k[:, :, t, :].unsqueeze(2) 
        v_t = v[:, :, t, :].unsqueeze(2) 
        a_t = alpha[:, :, t, :].unsqueeze(3) 
        i_s_t = input_scale[:, :, t, :].unsqueeze(3) 

        # Update
        kv_prod = torch.matmul(k_t.transpose(-2, -1), v_t) 
        state = a_t * state + i_s_t * kv_prod
        
        # Read
        y_t = torch.matmul(q_t, state) 
        outputs.append(y_t.squeeze(2))

    return torch.stack(outputs, dim=2)

# =============================================================================
# 4. Memoria Híbrida Coeus (RoPE + GQA)
# =============================================================================

class CoeusMemoryLayer(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.window_size

        # GQA Projections
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.gate_net = nn.Linear(config.n_embd, config.n_head) 
        self.norm_fusion = RMSNorm(config.n_embd)
        
        self.rotary = RotaryEmbedding(self.head_dim, config.block_size * 2)

    def forward(self, x: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        B, T, C = x.size()
        
        # 1. Proyecciones y Reshape
        # NOTA: .view en tensor output de Linear es seguro (contiguous)
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # [B, H, T, D]
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # [B, KV, T, D]
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # [B, KV, T, D]

        # 2. RoPE (Antes de expandir K para ahorrar cómputo)
        cos, sin = self.rotary(v) 
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA Branching
        # Expandimos K/V a full heads para usar en Recurrencia global (Operación per-head)
        # y en Local Attention.
        k = repeat_kv(k, self.n_rep) # [B, H, T, D]
        v = repeat_kv(v, self.n_rep) # [B, H, T, D]

        # 4. Gating (Alpha)
        gate_logits = self.gate_net(x).view(B, T, self.n_head, 1).transpose(1, 2)
        alpha = torch.sigmoid(gate_logits)

        # >>>> RAMA A: Global Neural Memory <<<<
        if inference_mode:
            y_global = recurrent_scan(q, k, v, alpha)
        else:
            # Parallel Scan (Log-Space)
            log_alpha = F.logsigmoid(gate_logits)
            log_input_scale = F.logsigmoid(-gate_logits)
            k_scaled = k * torch.exp(log_input_scale)

            log_alpha_cumsum = torch.cumsum(log_alpha, dim=2)
            log_decay_matrix = log_alpha_cumsum - log_alpha_cumsum.transpose(2, 3)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            log_decay_matrix = log_decay_matrix.masked_fill(mask, float("-inf"))
            decay_matrix = torch.exp(log_decay_matrix)
            
            # Atención global optimizada
            scores_global = torch.matmul(q, k_scaled.transpose(-2, -1))
            y_global = torch.matmul(scores_global * decay_matrix, v)

        # >>>> RAMA B: Local Block Attention <<<<
        # Aseguramos contigüidad antes del reshape complejo de bloques (Fix Profiler)
        q_l = q.transpose(1, 2).contiguous() # [B, T, H, D]
        k_l = k.transpose(1, 2).contiguous()
        v_l = v.transpose(1, 2).contiguous()
        
        pad_len = (self.window_size - (T % self.window_size)) % self.window_size
        T_padded = T + pad_len
        
        if pad_len > 0:
            q_l = F.pad(q_l, (0, 0, 0, 0, 0, pad_len))
            k_l = F.pad(k_l, (0, 0, 0, 0, 0, pad_len))
            v_l = F.pad(v_l, (0, 0, 0, 0, 0, pad_len))
        
        n_blocks = T_padded // self.window_size
        
        # Bloqueado seguro
        q_b = q_l.view(B, n_blocks, self.window_size, self.n_head, self.head_dim)\
               .permute(0, 1, 3, 2, 4)\
               .reshape(B * n_blocks, self.n_head, self.window_size, self.head_dim)
        
        k_b = k_l.view(B, n_blocks, self.window_size, self.n_head, self.head_dim)\
               .permute(0, 1, 3, 2, 4)\
               .reshape(B * n_blocks, self.n_head, self.window_size, self.head_dim)
               
        v_b = v_l.view(B, n_blocks, self.window_size, self.n_head, self.head_dim)\
               .permute(0, 1, 3, 2, 4)\
               .reshape(B * n_blocks, self.n_head, self.window_size, self.head_dim)

        # FlashAttention Local
        y_local_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)

        y_local = y_local_b.view(B, n_blocks, self.n_head, self.window_size, self.head_dim)\
                           .permute(0, 2, 1, 3, 4)\
                           .reshape(B, self.n_head, T_padded, self.head_dim)

        if pad_len > 0:
            y_local = y_local[:, :, :T, :] 

        # Fusión final
        y_out = y_global + y_local
        y_out = y_out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(self.norm_fusion(y_out))

# =============================================================================
# 5. NanoCoeus Wrapper (V2.5)
# =============================================================================
class Block(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CoeusMemoryLayer(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config) 

    def forward(self, x: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        # Residual connection
        x = x + self.attn(self.ln1(x), inference_mode=inference_mode)
        x = x + self.mlp(self.ln2(x))
        return x

class NanoCoeus(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe ha sido eliminado en favor de RoPE
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, inference_mode: bool = False):
        device = idx.device
        b, t = idx.size()
        
        x = self.wte(idx)
        x = self.drop(x)

        for block in self.h:
            x = block(x, inference_mode=inference_mode)
            
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None
        return logits, loss
