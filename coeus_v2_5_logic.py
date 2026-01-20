import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =============================================================================
# CONFIGURACIÓN COEUS V2.5 LOGIC-AWARE
# =============================================================================
@dataclass
class CoeusConfig:
    vocab_size: int = 50304
    dim: int = 512              # Embed dimension
    n_layers: int = 8           # Profundidad
    n_heads: int = 8            # Heads totales
    n_kv_heads: int = 2         # GQA: Grouped Query Attention (4 Query per KV)
    hidden_dim: int = 1536      # MLP width (3 * dim aprox)
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 1024     # Ajustado para 6GB VRAM de forma segura
    dropout: float = 0.1        # CRÍTICO: Prevención de Overfitting
    rope_theta: float = 10000.0
    
    # Logic Specifics
    gate_nonlinearity: bool = True # Usar MLP para la compuerta de memoria

# =============================================================================
# COMPONENTES AUXILIARES (RoPE, Norm)
# =============================================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return x_normed * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[2]]
    
    # Broadcast dimensions: [Batch, Heads, Seq, Dim/2]
    # Freqs shape: [Seq, Dim/2] -> Necesita unsqueeze para Batch y Heads
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1) 
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Implementación eficiente para GQA"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

# =============================================================================
# LOGIC AWARE RECURRENT SCAN (Memoria Global)
# =============================================================================
def parallel_scan_log(log_coeffs, log_values):
    """
    Scan paralelo estable numéricamente en log-space.
    h_t = a_t * h_{t-1} + x_t  -> log(h_t) log-sum-exp trick
    """
    a_star = torch.cumsum(log_coeffs, dim=-2)
    log_h0_plus_b_star = log_values - a_star
    log_h = a_star + torch.logcumsumexp(log_h0_plus_b_star, dim=-2)
    return torch.exp(log_h) # Regresamos al espacio lineal para la mezcla final

class NeuralGating(nn.Module):
    """
    GATE_NET AVANZADA: 
    Detecta cambios de contexto usando un MLP ligero en lugar de una proyección lineal.
    Esto permite "Atención Lógica": cerrar o abrir la memoria ante tokens clave (IF, THEN).
    """
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim // 2, bias=False)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(dim // 2, dim, bias=True)
        
        # Inicialización sesgada hacia la retención (Forget Gate cerca de 1)
        # Sigmoid(3.0) ~= 0.95. Queremos que por defecto RECUERDE, y aprenda a olvidar.
        nn.init.constant_(self.l2.bias, 3.0)

    def forward(self, x):
        return self.l2(self.act(self.l1(x)))

class GlobalRecurrence(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        
        # Neural Gating para "retención inteligente"
        self.gate_net = NeuralGating(self.dim)
        
        self.proj_in = nn.Linear(config.dim, config.dim * 2, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.c_norm = RMSNorm(config.dim)

    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Proyección Input y Gate
        # u: información nueva, g: control de olvido (log-space preparing)
        combined = self.proj_in(x)
        u, z = combined.chunk(2, dim=-1)
        
        # 2. Gating Mechanism (Logic Awareness)
        # Usamos logsigmoid para estabilidad numérica en el scan
        # gate_net(x) genera logits de olvido. 
        log_forget_gate = F.logsigmoid(self.gate_net(x)) 
        
        # 3. Parallel Scan (La magia O(N) paralela)
        # u se trata como el valor a inyectar (input)
        # scan calcula la acumulación histórica
        h_state = parallel_scan_log(log_forget_gate, u) # [B, T, C]
        
        # 4. Output projection
        out = self.proj_out(self.c_norm(h_state))
        return self.dropout(out)

# =============================================================================
# ATTENTION (Local Sliding Window + GQA)
# =============================================================================
class LocalAttention(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE
        xq, xk = apply_rotary_emb(xq.transpose(1, 2), xk.transpose(1, 2), freqs_cis)
        # Despues de RoPE están [B, H, T, D]
        
        # GQA: Repetir KV heads
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv.transpose(1, 2), self.n_rep) # [B, H, T, D]
        
        # Flash Attention (Windowed via Mask causal implicita en implementación normal, 
        # aqui dejamos full causal para simplificar en window size sliding es complejo sin custom kernel)
        out = F.scaled_dot_product_attention(
            xq, xk, xv, 
            is_causal=True, 
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

# =============================================================================
# BLOQUE HÍBRIDO TITAN (COEUS)
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CoeusBlock(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.local_attn = LocalAttention(config) # Sintaxis precisa
        self.global_rnn = GlobalRecurrence(config) # Contexto infinito lógico
        
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = FeedForward(config)
        
        # Alpha gate para mezclar atención local y global
        self.alpha = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x, freqs_cis):
        # Rama Paralela: Local + Global
        h = self.attn_norm(x)
        
        # Combinación ponderada aprendible
        attn_out = self.local_attn(h, freqs_cis) + (self.alpha * self.global_rnn(h))
        
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Coeus(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([CoeusBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight # Weight tying
        
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2, 
            config.rope_theta
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        
        freqs_cis = self.freqs_cis.to(x.device)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
