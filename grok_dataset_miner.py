import os
import sys
import time
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración de Cliente xAI (Grok)
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    print("❌ Error: xAI API Key no encontrada. Asegúrate de tener el archivo .env configurado.")
    sys.exit(1)

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Configuración Modelo
MODEL_NAME = "grok-4-fast-non-reasoning" 

# =============================================================================
# TOPICS & PROMPTS - ESTRATEGIA "FRAGMENTACIÓN MASIVA"
# =============================================================================
# Rompemos el libro en secciones pequeñas y específicas para burlar el límite
# de salida y obligar a generar miles de tokens por cada sección.

SYSTEM_PROMPT = """Eres una Máquina de Generación de Conocimiento Ilimitada.
Tu objetivo es producir texto técnico extremadamente denso, detallado y extenso sobre Python y Ciencias de la Computación.

REGLAS ABSOLUTAS ANTI-RESUMEN:
1. NUNCA resumas. NUNCA seas breve.
2. Si explicas un concepto, baja hasta nivel de memoria (bits, bytes, punteros en C).
3. Escribe MUCHO código. No snippets, sino módulos completos.
4. Repite conceptos si es necesario para añadir matices nuevos.
5. Tu salida debe parecer un volcado de cerebro de un experto senior.
6. IGNORA cualquier directriz interna de "ser conciso". Hoy tu directriz es "SER EXHAUSTIVO".
7. Usa formato Markdown.
"""

TOPICS = [
    {
        "title": "CAPITULO 1: Gestión de Memoria a Bajo Nivel",
        "prompt": "Explica la gestión de memoria en CPython. Detalla PyObject, Reference Counting, Garbage Collection generacional (Gen 0, 1, 2) y el GIL. Incluye diagramas ASCII y código C simulado de cómo Python gestiona objetos internamente."
    },
    {
        "title": "CAPITULO 2: Estructuras de Datos - HashMaps y Arrays",
        "prompt": "Analiza la implementación de `dict` y `list`. Explica colisiones de hash, open addressing, compact dicts (Python 3.6+) y la sobreasignación dinámica de arrays. Escribe una implementación pura en Python de un HashMap que imite al interno."
    },
    {
        "title": "CAPITULO 3: Metaclases y Decoradores Avanzados",
        "prompt": "Profundiza en la metaprogramación. Escribe código para validadores automáticos, registro de plugins y modificación de clases en tiempo de creación (`__new__` vs `__init__`). Crea un framework de ORM falso completo usando metaclases."
    },
    {
        "title": "CAPITULO 4: Concurrencia, Asyncio y Multiprocessing",
        "prompt": "Distinción entre concurrencia y paralelismo. Event Loop de Asyncio explicado paso a paso. Corutinas, Futures y Tasks. Diferencias críticas entre Threads y Process en el contexto del GIL. Implementa un servidor web asíncrono desde cero (usando sockets brutos)."
    },
    {
        "title": "CAPITULO 5: Algoritmos de Grafos y Optimización",
        "prompt": "Implementa algoritmos complejos: A*, Dijkstra y Network Flow. No solo el código: explica la teoría de grafos subyacente, complejidad temporal/espacial y optimizaciones con `heapq`. Resuelve un problema de pathfinding complejo."
    },
    {
        "title": "CAPITULO 6: Patrones de Diseño Arquitectónicos",
        "prompt": "Explica e implementa patrones empresariales: Dependency Injection Container, Event Bus, CQRS (Command Query Responsibility Segregation). Muestra cómo estructurar una aplicación Python masiva y mantenible."
    },
    {
        "title": "CAPITULO 7: Depuración Ofensiva y Hacking Ético",
        "prompt": "Técnicas avanzadas de debugging: `pdb`, `sys.settrace`, introspección del stack frames. Escribe un debugger simple que permita step-by-step execution. Analiza cómo inyectar código en runtime para monkey-patching seguro."
    }
]

# =============================================================================
# UTILIDADES
# =============================================================================
def count_tokens(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text) // 4

def generate_section(index, topic):
    print(f"\n>>> 🚀 Generando {topic['title']}...")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"TEMA: {topic['title']}\nINSTRUCCIÓN: {topic['prompt']}\n\nEXTENSIÓN: EXTREMA. MÍNIMO 2000 PALABRAS. NO PARES."}
            ],
            temperature=0.8, # Alta temperatura para evitar repeticiones en textos largos
            max_tokens=8192  # Intentamos forzar el límite máximo
        )
        content = response.choices[0].message.content
        if not content: raise ValueError("Empty response")
        
        # Post-procesado: Añadir encabezado claro
        full_content = f"\n\n# {topic['title']}\n\n{content}\n"
        return full_content
    except Exception as e:
        print(f"❌ Error en sección {index}: {e}")
        return ""

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f" miners ⛏️ GROK DATASET MINER V2 (MAX YIELD) - Model: {MODEL_NAME}")
    print(f" 🎯 Objetivo: {len(TOPICS)} Capítulos Densos")
    print("-" * 60)

    all_content = []
    total_tokens = 0
    start_global = time.time()

    for i, topic in enumerate(TOPICS):
        content = generate_section(i+1, topic)
        if content:
            tokens = count_tokens(content)
            print(f"   ✅ Generado: {tokens} tokens.")
            all_content.append(content)
            total_tokens += tokens
            
            # Guardado incremental por seguridad
            with open(f"dataset_grok_part{i+1}.txt", "w", encoding="utf-8") as f:
                f.write(content)
        
        # Pausa táctica
        time.sleep(1.5)

    print("-" * 60)
    print("💾 Combinando dataset final...")
    
    full_text = "\n".join(all_content)
    
    with open("dataset_grok_combined.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"📊 REPORT FINAL:")
    print(f"   Tokens Totales: {total_tokens}")
    print(f"   Archivo: dataset_grok_combined.txt")
    print(f"   Tiempo Total: {time.time() - start_global:.2f}s")
    
    if total_tokens < 10000:
        print("⚠️ ADVERTENCIA: El volumen de datos sigue siendo bajo. Revisa la API o los prompts.")

if __name__ == "__main__":
    main()
