import torch
import torch.nn.functional as F
import tiktoken
# Asegúrate de que el nombre del archivo del modelo coincida con el generado por el prompt
# Si el prompt genera 'coeus_v2_6.py', cambia esta línea
from coeus_v2_5_logic import Coeus, CoeusConfig 

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_PATH = 'coeus_logic_best.pt' # El archivo que guardará el entrenamiento nocturno
enc = tiktoken.get_encoding("gpt2")

def load_model():
    print(f"Loading checkpoint from {CKPT_PATH}...")
    # Fix for PyTorch 2.6+: weights_only=False required for custom classes like CoeusConfig
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Fallback para versiones anteriores de Torch
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # 1. Recuperar configuración original o usar defaults
    config = checkpoint.get('config', CoeusConfig())
    
    # 2. Instanciar Modelo
    model = Coeus(config).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Model Loaded. Trained for {checkpoint.get('iter', '?')} steps.")
    print(f"Best Loss: {checkpoint.get('best_loss', '?'):.4f}")
    return model

def generate(model, prompt, max_new_tokens=100, temperature=0.7):
    # Prepare Input
    input_ids = enc.encode(prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    print(f"\nPROMPT: {prompt}")
    print("COEUS: ", end='', flush=True)
    
    for _ in range(max_new_tokens):
        # Crop context if needed to fit block_size
        idx_cond = idx[:, -model.config.block_size:]
        
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                # Inference mode=True para usar el path optimizado si existe
                logits, _ = model(idx_cond, inference_mode=True)
            
            logits = logits[:, -1, :] # Last token
            
            # Sampling con Temperatura
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Print token (streaming)
            token = enc.decode(idx_next[0].tolist())
            print(token, end='', flush=True)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
    print("\n" + "-"*50)

def main():
    try:
        model = load_model()
    except FileNotFoundError:
        print(f"Error: No se encontró el checkpoint '{CKPT_PATH}'. Entrena primero.")
        return

    print(f"\n>>> INICIANDO TEST DE LÓGICA E INTELIGENCIA (Coeus v2.6) <<<")

    # TEST 1: Arithmetic Logic (Chain of Thought)
    # Buscamos ver si desglosa el problema
    print("\n[TEST 1: RAZONAMIENTO MATEMÁTICO]")
    generate(model, "User: Calculate 15 + 20 step by step.\nCoeus:", max_new_tokens=100, temperature=0.5)

    # TEST 2: Causal Reasoning
    # Lógica pura: A implica B. A sucede. ¿Qué pasa con B?
    print("\n[TEST 2: LÓGICA DEDUCTIVA]")
    generate(model, "Logic Exercise:\nIf it rains, the grass gets wet.\nIt is raining now.\nTherefore,", max_new_tokens=60, temperature=0.6)

    # TEST 3: Creative & Contextual Stability
    # Ver si alucina o mantiene el hilo narrativo
    print("\n[TEST 3: COHERENCIA NARRATIVA]")
    generate(model, "The scientist looked at the equation on the board and realized that", max_new_tokens=120, temperature=0.7)

if __name__ == "__main__":
    main()