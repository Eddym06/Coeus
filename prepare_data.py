import os
import requests

# CAMBIO DE PLAN PARA HACERLO INTERESANTE Y RÁPIDO:
# Vamos a bajar "Alicia en el País de las Maravillas" + "Frankenstein" + "Sherlock Holmes".
# Esto le dará variedad de vocabulario.

FILES = {
    "alice.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "holmes.txt": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt"
}

output_file = 'input.txt'

print(">>> Descargando Dataset 'Literary Mix'...")
with open(output_file, 'w', encoding='utf-8') as outfile:
    for name, url in FILES.items():
        print(f"   - Bajando {name}...")
        try:
            r = requests.get(url)
            r.encoding = 'utf-8'
            text = r.text
            # Limpieza básica de headers de Gutenberg
            start_idx = text.find("*** START OF")
            end_idx = text.find("*** END OF")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx]
            
            outfile.write(text + "\n\n")
        except Exception as e:
            print(f"Error bajando {name}: {e}")

print(f">>> Dataset listo en {output_file}. Tamaño: {os.path.getsize(output_file)/1024/1024:.2f} MB")
