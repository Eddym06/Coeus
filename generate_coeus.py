import torch
import os
from torch.nn import functional as F
from coeus import NanoCoeus, CoeusConfig 

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = 'coeus_ckpt_best.pt' # O 'coeus_ckpt.pt' si no se guardó el best

print(f"--- COEUS GENERATION PROTOCOL ---")
print(f"Loading brain from: {CKPT_PATH}")
print(f"Device: {DEVICE}")

# 1. Cargar el Checkpoint
if not os.path.exists(CKPT_PATH):
    # Fallback si no existe el _best
    print(f"Alerta: No se encontró {CKPT_PATH}, buscando checkpoint final...")
    CKPT_PATH = 'coeus_ckpt.pt'
    if not os.path.exists(CKPT_PATH):
         raise FileNotFoundError(f"No se encontró checkpoints. ¿Terminó el entrenamiento?")

checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
config = checkpoint['config'] # Recuperamos la config exacta del entrenamiento

# 2. Reconstruir el Modelo
model = NanoCoeus(config)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
model.eval() # IMPORTANTE: Modo evaluación (apaga dropout, etc)

print(f">> Model loaded successfully (Trained for {checkpoint['iter']} steps)")
print(f">> Best Validation Loss: {checkpoint.get('best_loss', 'N/A')}")

# 3. Tokenizer (Reconstruimos el mismo de Shakespeare)
# Nota: En un caso real, guardarías el tokenizer en un archivo json/pickle.
# Aquí lo reconstruimos rápido bajando el texto de nuevo (es muy pequeño).
import requests
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
try:
    text = requests.get(data_url).text
except:
    # Si falla la red, intenta leer local
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 4. Generación
print("\n>>> Escribe una frase para comenzar (o presiona Enter para empezar vacío):")
start_str = input("> ")
if start_str == "":
    start_str = "\n"

try:
    context = torch.tensor([encode(start_str)], dtype=torch.long, device=DEVICE)
except KeyError:
    print("Error: Usaste caracteres que no están en el vocabulario de Shakespeare. Usando vacío.")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    start_str = ""

print("\n>>> GENERANDO TEXTO (Coeus Thinking...):\n")
print(start_str, end='', flush=True)

# Generar 1000 caracteres
try:
    for _ in range(1000):
        # Recortar contexto si excede el block_size del modelo
        cond_idx = context[:, -config.block_size:]
        
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                logits, _ = model(cond_idx)
            
            logits = logits[:, -1, :] # Último token
            # Temperatura: < 1.0 es más conservador/preciso, > 1.0 es más creativo/loco
            probs = F.softmax(logits / 0.8, dim=-1) 
            
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
            
            # Imprimir caracter por caracter
            print(decode([idx_next.item()]), end='', flush=True)
except KeyboardInterrupt:
    print("\n\nInterrumpido por el usuario.")

print("\n\n>>> FIN DE LA TRANSMISIÓN.")