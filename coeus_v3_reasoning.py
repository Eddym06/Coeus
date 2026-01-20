import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =============================================================================
# CONFIGURACIÓN COEUS V3.0 (REASONING)
# =============================================================================
@dataclass
class CoeusConfig:
    vocab_size: int = 50304
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    reasoning_depth: int = 2    # Iteraciones del bloque recursivo

# =============================================================================
# COMPONENTES PROPIETARIOS V3 (Lógica y Razonamiento)
# =============================================================================

class ContrastiveLogicGate(nn.Module):
    """
    Componente A: Contrastive Gating.
    Decide dinámicamente si confiar en la Memoria Histórica (RNN) o en la Percepción Local (Attn).
    Reemplaza al escalar estático 'alpha'.
    """
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.proj_rnn = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_attn = nn.Linear(config.dim, config.dim, bias=False)
        
        # Red de confianza: Toma ambas señales y emite un score de mezcla per-channel
        self.confidence_net = nn.Sequential(
            nn.Linear(config.dim * 2, config.dim),
            nn.LayerNorm(config.dim), # Estabilidad extra
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
            nn.Sigmoid() # 0 = Puro RNN, 1 = Puro Attn
        )

    def forward(self, x_rnn, x_attn):
        # 1. Proyectar a espacio latente de decisión
        # (Opcional, pero ayuda a alinear espacios si RNN y Attn divergen)
        h_rnn = self.proj_rnn(x_rnn)
        h_attn = self.proj_attn(x_attn)
        
        # 2. Análisis de Conflicto
        decision_input = torch.cat([h_rnn, h_attn], dim=-1)
        gate = self.confidence_net(decision_input)
        
        # 3. Mezcla Dinámica Canal por Canal
        return (1 - gate) * x_rnn + gate * x_attn

class FeedForward(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class DeepReasoningBlock(nn.Module):
    """
    Componente B: Recursive Reasoning Step.
    Aplica el mismo FFN múltiples veces para refinar el estado (Universal Transformer style).
    Simula "tiempo para pensar".
    """
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.processor = FeedForward(config)
        self.iterations = config.reasoning_depth
        self.norm = RMSNorm(config.dim) # Pre-Norm para estabilidad recursiva

    def forward(self, x):
        reasoning_state = x
        for _ in range(self.iterations):
            # Residual connection interna en cada paso de pensamiento
            delta = self.processor(self.norm(reasoning_state))
            reasoning_state = reasoning_state + delta
        return reasoning_state

# =============================================================================
# COMPONENTES CORE (RoPE, Norm, Scan)
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
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1) 
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1: return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

# Parallel Scan Log-Space (Estabilizado)
def parallel_scan_log(log_coeffs, log_values):
    a_star = torch.cumsum(log_coeffs, dim=-2)
    log_h0_plus_b_star = log_values - a_star
    log_h = a_star + torch.logcumsumexp(log_h0_plus_b_star, dim=-2)
    return torch.exp(log_h)

class GlobalRecurrence(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.dim = config.dim
        
        # MEJORA: Gating de canal completo SIN cuello de botella
        self.gate_net = nn.Linear(config.dim, config.dim) 
        # Inicialización fuerte hacia "Recordar" (bias=2.0 -> sigmoid ~0.88)
        nn.init.constant_(self.gate_net.bias, 2.0)
        
        self.proj_in = nn.Linear(config.dim, config.dim * 2, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.c_norm = RMSNorm(config.dim)

    def forward(self, x):
        combined = self.proj_in(x)
        u, z = combined.chunk(2, dim=-1)
        
        # Log-Space Gating (Full Channel)
        log_forget_gate = F.logsigmoid(self.gate_net(x))
        
        h_state = parallel_scan_log(log_forget_gate, u)
        out = self.proj_out(self.c_norm(h_state))
        return self.dropout(out)

class LocalAttention(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
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

        xq, xk = apply_rotary_emb(xq.transpose(1, 2), xk.transpose(1, 2), freqs_cis)
        
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv.transpose(1, 2), self.n_rep) 
        
        out = F.scaled_dot_product_attention(
            xq, xk, xv, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))

# =============================================================================
# BLOQUE HÍBRIDO V3 (Reasoning)
# =============================================================================
class CoeusBlock(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        
        # Ramas
        self.local_attn = LocalAttention(config)
        self.global_rnn = GlobalRecurrence(config)
        
        # Componente A: Logic Gate (Arbitro)
        self.logic_gate = ContrastiveLogicGate(config)
        
        self.ffn_norm = RMSNorm(config.dim)
        # Componente B: Recursive Reasoning
        self.reasoning = DeepReasoningBlock(config)

    def forward(self, x, freqs_cis):
        # 1. ATENCIÓN HÍBRIDA
        h = self.attn_norm(x)
        
        # Calcular ramas en paralelo
        out_local = self.local_attn(h, freqs_cis)
        out_global = self.global_rnn(h)
        
        # Fusión Inteligente (Contrastive)
        # El modelo decide cuánto pesa la historia vs el contexto local
        mixed_attn = self.logic_gate(out_global, out_local)
        
        x = x + mixed_attn
        
        # 2. RAZONAMIENTO PROFUNDO (Universal Transformer)
        # Itera el FFN varias veces para "pensar"
        x = x + self.reasoning(self.ffn_norm(x))
        return x

class Coeus(nn.Module):
    def __init__(self, config: CoeusConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([CoeusBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2, 
            config.rope_theta
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
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
