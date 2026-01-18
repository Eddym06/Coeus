
import os
import time
import math
import pickle
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from coeus import NanoCoeus, CoeusConfig 

# Activar TF32 globalmente para obtener velocidad gratis en Ampere/Ada
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# 1. CONFIGURACIÓN DEL ENTRENAMIENTO OPIMIZADA (RTX 4050 8GB)
# =============================================================================
# Estrategia: "VRAM-First". Evitamos desbordar a RAM del sistema (Shared Memory)
# Batch size pequeño compensado con Gradient Accumulation implícito (si quisiéramos)
BATCH_SIZE = 2       # VRAM-Safe Mode: Bajamos a 2 para asegurar estabilidad con contexto 2048.
BLOCK_SIZE = 2048    # Contexto 
MAX_ITERS = 5000
EVAL_INTERVAL = 250  # Evaluar más frecuente al principio
LEARNING_RATE = 6e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Preferimos bfloat16 (Ampere+) si está disponible, es más estable numéricamente que float16
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"--- COEUS TRAINING PROTOCOL ---")
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")
print(f"Precision: {DTYPE}")
print(f"Target Memory: Strict VRAM (Avoid Shared RAM)")
print("-" * 40)

# =============================================================================
# 2. DATASET: TINY SHAKESPEARE
# =============================================================================
file_path = 'input.txt'
if not os.path.exists(file_path):
    print(">> Downloading TinyShakespeare dataset...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizer simple (caracteres)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/Val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# =============================================================================
# 3. INICIALIZAR MODELO COEUS
# =============================================================================
config = CoeusConfig()
config.vocab_size = vocab_size 
config.block_size = BLOCK_SIZE
config.n_layer = 6
config.n_head = 6
config.n_embd = 384
config.window_size = 128 

model_raw = NanoCoeus(config)
model_raw.to(DEVICE)

print(f"Model Parameters: {sum(p.numel() for p in model_raw.parameters())/1e6:.2f}M")

# COMPILATION SAFETY NET (Windows Fix)
print(">> Attempting to compile Coeus model...")
model_opt = model_raw # Default fallback
try:
    # Usamos reduce-overhead para bucles de entrenamiento rápidos
    # Solo compilamos para la ejecución, mantenemos model_raw para state_dict
    compiled = torch.compile(model_raw, mode='reduce-overhead')
    
    # Test run rapidísimo para verificar compatibilidad (Triton check)
    dummy_x, dummy_y = get_batch('train')
    # Solo 2 tokens para no gastar tiempo
    dummy_x = dummy_x[:, :2] 
    dummy_y = dummy_y[:, :2]
    
    with torch.autocast(device_type=DEVICE, dtype=DTYPE):
        compiled(dummy_x, dummy_y)
        
    model_opt = compiled
    print(">> ✅ Compilation SUCCESS. Training will be blazing fast.")
except Exception as e:
    print(f">> ⚠️ Compilation skipped (Common on Windows w/o Triton): {e}")
    print(">> Fallback to EAGER mode. Still fast thanks to TF32 & AMP.")

optimizer = torch.optim.AdamW(model_raw.parameters(), lr=LEARNING_RATE)

# Estimador de Loss robusto
@torch.no_grad()
def estimate_loss():
    out = {}
    model_raw.eval() # Ponemos el modelo base en eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                # Usamos model_opt para inferencia eficiente si es posible,
                # pero model_raw es más seguro si model_opt tiene graph caches raros para batch sizes variables
                # Para seguridad absoluta en eval con tamaños fijos, usamos model_opt
                logits, loss = model_opt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_raw.train() # Volvemos a train
    return out

# =============================================================================
# 4. BUCLE DE ENTRENAMIENTO
# =============================================================================
print("\n>>> STARTING TRAINING 🚀")
print("    (Press Ctrl+C to stop manually if needed)")
t0 = time.time()
best_val_loss = float('inf')

try:
    for iter in range(MAX_ITERS):
        # 1. Obtener batch
        xb, yb = get_batch('train')

        # 2. Forward & Backward con Mixed Precision
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            logits, loss = model_opt(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 3. Clip gradientes (CRUCIAL para Coeus/RNNs)
        # Previene que el estado de memoria explote numéricamente
        torch.nn.utils.clip_grad_norm_(model_raw.parameters(), 1.0)
        
        optimizer.step()

        # 4. Logs y Checkpoints
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss()
            dt = time.time() - t0
            
            # Monitor VRAM
            if DEVICE == 'cuda':
                vram = torch.cuda.max_memory_allocated() / 1e9 #(GB)
                mem_str = f"| VRAM: {vram:.1f}GB"
            else:
                mem_str = ""
            
            print(f"Step {iter}: Train {losses['train']:.4f}, Val {losses['val']:.4f} | Time: {dt:.2f}s {mem_str}")
            
            # Guardar mejor modelo
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter > 0:
                    checkpoint = {
                        'model': model_raw.state_dict(), # Guardamos state_dict, no el modelo compilado
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter': iter,
                        'best_loss': best_val_loss
                    }
                    torch.save(checkpoint, 'coeus_ckpt_best.pt')
                    # print("   (Checkpoint saved)")
                
            t0 = time.time()

except KeyboardInterrupt:
    print("\n🛑 Training stopped manually.")

# =============================================================================
# 5. GENERACIÓN DE TEXTO (INFERENCIA)
# =============================================================================
print("\n>>> GENERATING SAMPLE TEXT...")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Start token (0)
model_raw.eval()

# Nota: La implementación actual usa Parallel Scan, lo cual es ineficiente para 
# generación autoregresiva token a token (O(N^2) recompute vs O(1) RNN state).
# Para el prototipo training-first esto es aceptable.
print("Generating (can be slow without RNN-mode inference)...")

try:
    for _ in range(500):
        # Recortar contexto si excede block_size
        idx_cond = context[:, -BLOCK_SIZE:]
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                logits, _ = model_raw(idx_cond) # Usamos raw para inferencia flexible
            logits = logits[:, -1, :] # Último token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)

    print("-" * 50)
    print(decode(context[0].tolist()))
    print("-" * 50)

except Exception as e:
    print(f"Generation error: {e}")
