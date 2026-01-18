import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# 1. Configuración Hardware-Aware (Update V2.5)
# =============================================================================
@dataclass
class CoeusConfig:
    block_size: int = 2048       # Contexto máximo para entrenamiento
    vocab_size: int = 50257      
    n_layer: int = 6             
    n_head: int = 6              
    n_kv_head: int = 2           # GQA: Cabezas de Key/Value (6/2 = 3x repetición)
    n_embd: int = 384            
    window_size: int = 128       # Tamaño de bloque local
    dropout: float = 0.0         

# =============================================================================
# 2. Componentes Optimizados
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        # SwiGLU: (Linear * Sigmoid) * Linear
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
        # x: [B, T, H, D] o similar. Lo importante es dimension T.
        # Asumimos que x viene transpuesto o leemos shape T correcto.
        # Aquí usaremos el shape[2] porque llamamos con V transpuesta [B, KV, T, D]
        t = torch.arange(x.shape[2], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [T, D]
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, T, D]
    # cos, sin: [T, D] -> Unsqueeze para broadcasting [1, 1, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# =============================================================================
# 3. Kernel Recurrente JIT (O(1) Memory Inference)
# =============================================================================
@torch.jit.script
def recurrent_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Ejecuta la recurrencia O(N) -> O(1) memoria.
    M_t = alpha_t * M_{t-1} + (1-alpha_t) * (K_t^T V_t)
    y_t = Q_t * M_t
    """
    B, H, T, D = q.shape
    # Estado de memoria inicial [B, H, D, D]
    state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    
    # Pre-computar input scale (1 - alpha)
    input_scale = 1.0 - alpha

    for t in range(T):
        # [B, H, 1, D]
        q_t = q[:, :, t, :].unsqueeze(2)
        k_t = k[:, :, t, :].unsqueeze(2) 
        v_t = v[:, :, t, :].unsqueeze(2) 
        a_t = alpha[:, :, t, :].unsqueeze(3) 
        i_s_t = input_scale[:, :, t, :].unsqueeze(3) 

        # Actualizar Memoria: M_t
        kv_prod = torch.matmul(k_t.transpose(-2, -1), v_t) 
        state = a_t * state + i_s_t * kv_prod
        
        # Leer Memoria: y = q * M_t
        y_t = torch.matmul(q_t, state) 
        outputs.append(y_t.squeeze(2))

    return torch.stack(outputs, dim=2) # [B, H, T, D]

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    B, n_kv_head, T, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_head, n_rep, T, head_dim)
        .reshape(B, n_kv_head * n_rep, T, head_dim)
    )

# =============================================================================
# 4. Módulo de Memoria Híbrido: COEUS v2.5 (RoPE + GQA)
# =============================================================================

class CoeusMemoryLayer(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.window_size = config.window_size

        # GQA: Proyecciones separadas
        # Query tiene todas las cabezas
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        # Key y Value tienen menos cabezas (GQA)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # El gating es específico para cada Query Head, por lo tanto usa n_head
        self.gate_net = nn.Linear(config.n_embd, config.n_head) 
        self.norm_fusion = RMSNorm(config.n_embd)
        
        # Rotary Embeddings
        # Se aplican a Q y K. Inician con dims de head_dim.
        self.rotary = RotaryEmbedding(self.head_dim, config.block_size * 2)

    def forward(self, x: torch.Tensor, inference_mode: bool = False) -> torch.Tensor:
        B, T, C = x.size()
        
        # 1. Proyecciones GQA
        # Q -> [B, T, H, D] -> Transpose -> [B, H, T, D]
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # K, V -> [B, T, KV_H, D] -> Transpose -> [B, KV_H, T, D]
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # 2. Rotary Positional Embeddings (RoPE)
        # Calculamos freq cis. Pasamos 'v' solo para obtener T y device correctos.
        cos, sin = self.rotary(v) 
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA Expansion (Repeat K/V)
        # Antes de entrar a Recurrencia o Atención Local, necesitamos que K/V tengan n_head igual a Q
        # [B, KV_H, T, D] -> [B, H, T, D]
        k = repeat_kv(k, self.n_rep) 
        v = repeat_kv(v, self.n_rep)

        # 4. Gating (Alpha)
        gate_logits = self.gate_net(x).view(B, T, self.n_head, 1).transpose(1, 2)
        alpha = torch.sigmoid(gate_logits)

        # ---------------------------------------------------------------------
        # RAMA A: Global Neural Memory (Linear Recurrence)
        # ---------------------------------------------------------------------
        if inference_mode:
            # Modo recurrente token a token
            y_global = recurrent_scan(q, k, v, alpha)
        else:
            # Modo paralelo de entrenamiento
            log_alpha = F.logsigmoid(gate_logits)
            log_input_scale = F.logsigmoid(-gate_logits)
            k_scaled = k * torch.exp(log_input_scale)

            log_alpha_cumsum = torch.cumsum(log_alpha, dim=2)
            log_decay_matrix = log_alpha_cumsum - log_alpha_cumsum.transpose(2, 3)
            # Máscara causal triangular
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            log_decay_matrix = log_decay_matrix.masked_fill(mask, float("-inf"))
            decay_matrix = torch.exp(log_decay_matrix)
            
            # Atención Lineal: (Q @ K^T) @ V
            scores_global = torch.matmul(q, k_scaled.transpose(-2, -1))
            y_global = torch.matmul(scores_global * decay_matrix, v)

        # ---------------------------------------------------------------------
        # RAMA B: Local Block Attention (Sliding Window / Block)
        # ---------------------------------------------------------------------
        # Preparamos tensores para vista en bloques: necesitan ser [B, T, H, D] contiguous
        q_l = q.transpose(1, 2).contiguous()
        k_l = k.transpose(1, 2).contiguous()
        v_l = v.transpose(1, 2).contiguous()
        
        pad_len = (self.window_size - (T % self.window_size)) % self.window_size
        T_padded = T + pad_len
        
        # Padding si es necesario
        if pad_len > 0:
            q_l = F.pad(q_l, (0, 0, 0, 0, 0, pad_len))
            k_l = F.pad(k_l, (0, 0, 0, 0, 0, pad_len))
            v_l = F.pad(v_l, (0, 0, 0, 0, 0, pad_len))
        
        n_blocks = T_padded // self.window_size
        
        # Reshape Mágico: [B, Blocks, Window, H, D] -> Flash Attention espera batch unificado
        # Transformamos a [B*Blocks, H, Window, D]
        def to_block_view(tensor):
            return tensor.view(B, n_blocks, self.window_size, self.n_head, self.head_dim)\
                         .permute(0, 1, 3, 2, 4)\
                         .reshape(B * n_blocks, self.n_head, self.window_size, self.head_dim)

        q_b = to_block_view(q_l)
        k_b = to_block_view(k_l)
        v_b = to_block_view(v_l)

        # Computo local optimizado (SDPA)
        y_local_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)

        # Reconstrucción
        y_local = y_local_b.view(B, n_blocks, self.n_head, self.window_size, self.head_dim)\
                           .permute(0, 2, 1, 3, 4)\
                           .reshape(B, self.n_head, T_padded, self.head_dim)

        if pad_len > 0:
            y_local = y_local[:, :, :T, :] 

        # ---------------------------------------------------------------------
        # Fusión Híbrida
        # ---------------------------------------------------------------------
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
        x_attn = self.attn(self.ln1(x), inference_mode=inference_mode)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x

class NanoCoeus(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # ROPE-UPDATE: Eliminamos wpe (Absolute Position Embedding)
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
        
        # Solo Embeddings de Token. La posición se inyecta vía RoPE en el self-attention.
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
