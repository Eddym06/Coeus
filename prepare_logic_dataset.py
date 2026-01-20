import random
import os

# =============================================================================
# CONFIGURACION
# =============================================================================
LOGIC_RATIO = 0.6  # 60% Lógica
OUTPUT_FILE = "logic_mix_large.txt" # Nuevo nombre para diferenciar
INPUT_LIT_FILE = "input.txt" 
OUTPUT_SIZE_CHARS = 15 * 1024 * 1024 # Objetivo: ~15 MB de texto

# =============================================================================
# GENERADORES DE CHAIN OF THOUGHT (CoT) AVANZADOS
# =============================================================================

def gen_arithmetic_cot():
    """Genera sumas/multiplicaciones explicadas paso a paso."""
    ops = ['+', '-', '*']
    op = random.choice(ops)
    
    if op == '+':
        a, b = random.randint(100, 9999), random.randint(100, 9999)
        res = a + b
        text = f"User: Calculate {a} + {b}.\n"
        text += "Coeus: Let's break this down step by step using column addition logic.\n"
        text += f"1. Write {a} above {b}.\n"
        text += f"2. Sum units: {a%10} + {b%10} = {(a%10)+(b%10)}.\n"
        text += f"3. Sum tens: {(a//10)%10}0 + {(b//10)%10}0 = {((a//10)%10 + (b//10)%10)*10}.\n"
        text += f"4. Processing carries internally... The sum is {res}.\n"
        text += f"Therefore, {a} + {b} = {res}.\n\n"
        return text

    elif op == '-':
        a, b = random.randint(1000, 9999), random.randint(10, 999)
        res = a - b
        text = f"User: Solve {a} - {b}.\n"
        text += "Coeus: Subtraction reasoning:\n"
        text += f"We start with {a} and remove {b}.\n"
        text += f"Approximation: {a} - {b} is close to {a - (b//100)*100}.\n"
        text += f"Exact calculation: {a} minus {b} results in {res}.\n"
        text += f"Answer: {res}.\n\n"
        return text
    
    elif op == '*':
        a, b = random.randint(5, 50), random.randint(2, 20)
        res = a * b
        text = f"Problem: What is {a} times {b}?\n"
        text += "Reasoning: Multiplication is repeated addition.\n"
        text += f"Step 1: Decompose {b} into parts if needed.\n"
        text += f"Step 2: {a} multiplied by {b} equals {res}.\n"
        text += f"Final Answer: {res}.\n\n"
        return text
    return ""

def gen_symbolic_logic():
    """Silogismos y lógica formal."""
    entities = ['Socrates', 'The robot', 'The algorithm', 'A triangle', 'The input']
    categories = ['mortal', 'deterministic', 'geometry', 'data', 'optimized']
    
    e = random.choice(entities)
    c = random.choice(categories)
    
    modes = [
        f"Premise 1: All {c} objects are valid.\nPremise 2: {e} is {c}.\nConclusion: Therefore, {e} is valid.",
        f"Rule: If X is {c}, then X must process.\nObservation: {e} is {c}.\nDeduction: {e} must process.",
        f"Statement: {e} cannot be False and True at the same time.\nCheck: {e} is True.\nResult: {e} is not False."
    ]
    return f"Logic Task:\n{random.choice(modes)}\n\n"

def gen_coding_logic():
    """Pseudocódigo explicado."""
    tasks = [
        ("sort a list", "iterate through elements and swap if predictable."),
        ("find max value", "initialize max to -infinity, update if current > max."),
        ("invert string", "read from end to start.")
    ]
    t, r = random.choice(tasks)
    return f"User: How do you {t}?\nCoeus: Algorithm:\n1. Start process.\n2. We must {r}\n3. Return result.\nEnd.\n\n"

def gen_logic_block():
    r = random.random()
    if r < 0.4: return gen_arithmetic_cot()
    elif r < 0.7: return gen_symbolic_logic()
    else: return gen_coding_logic()

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(">>> PREPARING 'LOGIC MIX LARGE' DATASET (Target: 15MB)...")
    
    # 1. Cargar Literatura
    lit_text = ""
    try:
        with open(INPUT_LIT_FILE, 'r', encoding='utf-8') as f:
            lit_text = f.read()
    except FileNotFoundError:
        lit_text = "Alice was logical. " * 5000 # Placeholder

    # Repertir literatura si es muy corta para el mix
    while len(lit_text) < (OUTPUT_SIZE_CHARS * (1 - LOGIC_RATIO)):
        lit_text += "\n" + lit_text

    print(f" Literature Base: {len(lit_text)/1024/1024:.2f} MB")
    
    # 2. Generar
    logic_data = []
    current_len = 0
    target_logic = int(OUTPUT_SIZE_CHARS * LOGIC_RATIO)
    
    print(" Generating Logic/CoT data (this helps prevent overfitting)...")
    while current_len < target_logic:
        block = gen_logic_block()
        logic_data.append(block)
        current_len += len(block)
    
    logic_text = "".join(logic_data)
    
    # 3. Mezclar
    chunk_size = 4096 # Bloques mas grandes de contexto
    lit_chunks = [lit_text[i:i+chunk_size] for i in range(0, len(lit_text), chunk_size)]
    logic_chunks = [logic_text[i:i+chunk_size] for i in range(0, len(logic_text), chunk_size)]
    
    final_mix = []
    while lit_chunks or logic_chunks:
        if random.random() < LOGIC_RATIO and logic_chunks:
            final_mix.append(logic_chunks.pop(0))
        elif lit_chunks:
            final_mix.append(lit_chunks.pop(0))
        elif logic_chunks:
             final_mix.append(logic_chunks.pop(0))
             
    full_text = "".join(final_mix)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_text)
        
    print(f"✅ DATASET READY: {OUTPUT_FILE}")
    print(f"   Final Size: {len(full_text)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
