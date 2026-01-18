import torch
import torch.nn.functional as F
from coeus import NanoCoeus, CoeusConfig
import time
import psutil
import sys
import os

# =============================================================================
# HARDWARE SETUP
# =============================================================================
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# =============================================================================
# SAFETY MONITORING
# =============================================================================
def check_system_health(threshold_percent=90.0):
    """
    Checks system RAM usage. If it exceeds threshold, raises an exception to abort.
    """
    mem = psutil.virtual_memory()
    if mem.percent > threshold_percent:
        raise RuntimeError(f"⚠️ SISTEMA EN PELIGRO: RAM al {mem.percent}%! Abortando para evitar crash.")
    return mem.percent

# =============================================================================
# MODEL SETUP
# =============================================================================
# Usamos la config por defecto de Coeus (V2)
config = CoeusConfig()
model = NanoCoeus(config).to(device)

# Load weights if available (optional for stress test, but good for validity)
ckpt_path = 'coeus_ckpt_best.pt'
if os.path.exists(ckpt_path):
    print(f"Cargando checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=False) # strict=False por si acaso cambiaron nombres en V2
else:
    print("⚠️ No se encontró checkpoint, usando pesos aleatorios inicializados.")

model.eval()

# =============================================================================
# 30K TOKEN STRESS TEST
# =============================================================================
TARGET_TOKENS = 30000
INITIAL_TOKENS = 64 # Semilla pequeña

print("\n🚀 INICIANDO STRESS TEST: 30,000 TOKENS")
print(f"Protección de RAM activa (Umbral: 90%)")
print("-" * 50)

# Dummy input
x = torch.randint(0, config.vocab_size, (1, INITIAL_TOKENS)).to(device)

# VRAM Baseline
torch.cuda.synchronize()
vram_start = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM Inicial: {vram_start:.2f} GB")

start_time = time.time()

try:
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # En V2 (Block Attention), pasar todo el contexto de golpe podría seguir siendo pesado si no es chunked en atención, 
            # pero el modelo forward() maneja la secuencia completa.
            # Sin embargo, generate token a token es muy lento para 30k.
            # Haremos saltos grandes para simular carga.
            
            # Vamos a simular creando tensores de entrada crecientes para ver si el forward pass aguanta.
            # No generaremos autoregresivamente 30k tokens uno a uno (tardaría horas),
            # sino que probaremos hacer un forward pass con contexto de tamaños crecientes: 1k, 5k, 10k, 15k, 30k.
            
            checkpoints = [1024, 5000, 10000, 15000, 20000, 30000]
            
            for seq_len in checkpoints:
                print(f"\n🔍 Probando contexto: {seq_len} tokens...")
                
                # Check RAM before alloc
                ram_usage = check_system_health()
                
                # Crear input dummy del tamaño deseado
                x_test = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
                
                t0 = time.time()
                # Activamos el modo inferencia recurrente O(1) memoria
                logits, _ = model(x_test, inference_mode=True)
                torch.cuda.synchronize()
                dt = time.time() - t0
                
                # Stats
                vram_curr = torch.cuda.memory_allocated() / 1024**3
                vram_peak = torch.cuda.max_memory_allocated() / 1024**3
                
                print(f"✅ Éxito {seq_len} tokens.")
                print(f"   ⏱️ Tiempo Forward: {dt*1000:.2f} ms")
                print(f"   💾 VRAM Actual: {vram_curr:.2f} GB")
                print(f"   📈 VRAM Pico: {vram_peak:.2f} GB")
                print(f"   🧠 System RAM: {ram_usage}%")
                
                # Limpiar cache para la siguiente prueba
                del x_test, logits
                torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("🏆 PRUEBA SUPERADA: Coeus V2 manejó 30,000 tokens sin explotar.")
    print("=" * 50)

except RuntimeError as e:
    print("\n" + "!" * 50)
    print(f"❌ ERROR/ABORTADO: {e}")
    # Si es CUDA OOM, pytorch lo lanza como RuntimeError
    if "CUDA out of memory" in str(e):
        print("💡 Diagnóstico: OOM de Video. La VRAM se llenó.")
    print("!" * 50)

except Exception as e:
    print(f"\n❌ Error inesperado: {e}")

finally:
    total_time = time.time() - start_time
    print(f"\nTiempo total del test: {total_time:.2f}s")
