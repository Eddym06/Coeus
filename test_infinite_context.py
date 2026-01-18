import torch
import time
from coeus import NanoCoeus, CoeusConfig

# Configuración "Infinito"
config = CoeusConfig()
config.block_size = 4096 # ¡4K Contexto!
config.n_embd = 384
config.n_head = 6
config.n_layer = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Testing Infinite Context on {device}...")

# Instanciar modelo (sin entrenar, solo estructura)
model = NanoCoeus(config)
model.to(device)
model.eval() # Modo evaluación (ahorra mucha memoria)

# Crear input masivo (Batch 1, Secuencia 4096)
x = torch.randint(0, 50257, (1, 4096), device=device)

print(f"Allocated Tensor: [1, 4096]. Running Forward Pass...")

try:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    
    with torch.no_grad(): # Sin gradientes = Memoria Lineal real
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, _ = model(x)
            
    t1 = time.time()
    mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"✅ SUCCESS!")
    print(f"Time: {(t1-t0)*1000:.2f}ms")
    print(f"Peak VRAM: {mem:.2f} GB")
    print("Conclusion: La arquitectura soporta 4096 tokens en Inferencia.")
    
except Exception as e:
    print(f"❌ FAILED: {e}")