import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Importar el nuevo modelo refinado V3
from coeus_v3_reasoning import Coeus, CoeusConfig

# =============================================================================
# CONFIGURACIÓN "SMART TRAIN NIGHTLY" (6 HOURS ROBUST)
# =============================================================================
MASTER_SEED = 1337
torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed(MASTER_SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
torch.set_float32_matmul_precision('high') 

# Hyperparameters REFINED for Long Run VI (Reasoning Edition)
BATCH_SIZE = 4        
GRAD_ACCUM_STEPS = 16 # Effective Batch 64
BLOCK_SIZE = 1024     
MAX_ITERS = 5000      
LEARNING_RATE = 1.6e-4  # Lowered slightly for stability over long run
MIN_LR = 1.6e-5
WARMUP_ITERS = 500
PATIENCE = 20         # Aumentado para V3 (Deep Reasoning necesita mas tiempo de settling)
MIN_ITERS_BEFORE_STOP = 2000 # Obligar a entrenar al menos 2000 pasos (aprox 2.5h)

DATA_FILE = 'dataset_grok_combined.txt' # Dataset grande
CKPT_BEST = 'coeus_reasoning_best.pt'
CKPT_LAST = 'coeus_reasoning_last.pt'

# =============================================================================
# UTILS
# =============================================================================
def get_batch(data_tensor, block_size, batch_size, device):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it > MAX_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} need to be generated first.")
        return

    # 1. Prepare Data
    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoder = tiktoken.get_encoding("gpt2")
    data = torch.tensor(encoder.encode(text), dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train Tokens: {len(train_data)} | Val Tokens: {len(val_data)}")

    # 2. Init Model
    config = CoeusConfig()
    config.max_seq_len = BLOCK_SIZE
    print(f"Initializing COEUS V3.0 REASONING (LogicGate + DeepRecursive)...")
    model = Coeus(config).to(DEVICE)
    
    # EAGER MODE ONLY (Stability)
    print("Running in EAGER mode (Safe for Nightly run)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    t0 = time.time()
    
    print(">>> STARTING ROBUST NIGHTLY TRAINING <<<")
    print(f"    Effective Batch: 64 | Patience: {PATIENCE} | Min Steps: {MIN_ITERS_BEFORE_STOP}")
    
    for iter in range(MAX_ITERS + 1):
        # Update LR
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Gradient Accumulation
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        
        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=DTYPE):
                _, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS 
            loss.backward()
            acc_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if iter % 100 == 0:
            losses = estimate_loss(model, train_data, val_data)
            dt = time.time() - t0
            t0 = time.time()
            print(f"Step {iter}: Train {losses['train']:.4f}, Val {losses['val']:.4f} | LR {lr:.2e} | Time {dt:.2f}s")
            
            # Save Checkpoint Logic
            # Save LAST always
            ckpt_last = {'model': model.state_dict(), 'config': config, 'iter': iter, 'loss': losses['val']}
            torch.save(ckpt_last, CKPT_LAST)

            # Save BEST
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                torch.save(ckpt_last, CKPT_BEST)
                print(f"  💾 BEST checkpoint saved ({best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE and iter > MIN_ITERS_BEFORE_STOP:
                    print(f"\n✋ EARLY STOPPING (Stable). Val loss rose for {PATIENCE} cycles after min steps.")
                    break
        
        # Monitor Intelligence
        if iter % 1000 == 0 and iter > 0:
             print("\n🔍 LOGIC CHECK:")
             # Simple inline generation
             ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
             for _ in range(40):
                with torch.no_grad():
                    l, _ = model(ctx[:, -BLOCK_SIZE:])
                    next_tk = torch.multinomial(F.softmax(l[:, -1, :], dim=-1), 1)
                    ctx = torch.cat((ctx, next_tk), dim=1)
             print(encoder.decode(ctx[0].tolist()))
             print("-" * 30)

    print(">>> TRAINING COMPLETE. <<<")

if __name__ == "__main__":
    main()
