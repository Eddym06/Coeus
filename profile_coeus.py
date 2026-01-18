import torch
import torch.nn as nn
from coeus import NanoCoeus, CoeusConfig
from torch.profiler import profile, record_function, ProfilerActivity

# 1. Configuración
config = CoeusConfig()
config.block_size = 1024 # Tamaño estándar de entrenamiento
config.n_head = 6
config.n_kv_head = 2     # ¡Probando GQA! (Asegúrate de que coeus.py ya tenga GQA si usas esto)
# Si aún no has aplicado el prompt de arriba, comenta n_kv_head y usa el coeus actual.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NanoCoeus(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

# Datos dummy
x = torch.randint(0, config.vocab_size, (4, config.block_size), device=device)
y = torch.randint(0, config.vocab_size, (4, config.block_size), device=device)

print(f"--- COEUS PERFORMANCE PROFILER ---")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("Warming up GPU...")

# Warm-up (para cargar kernels CUDA y que no salgan en el reporte)
for _ in range(3):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Profiling started... (This takes ~10 seconds)")

# 2. Ejecutar Profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_training_step"):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 3. Reporte
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Exportar para visualizar en Chrome (chrome://tracing)
prof.export_chrome_trace("trace.json")
print("\n✅ Análisis guardado en 'trace.json'. Puedes abrirlo en Chrome para ver el timeline.")