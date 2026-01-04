"""
Test semplificato - Verifica la struttura del codice senza dipendenze esterne
"""

import sys
import os

print("="*60)
print("TEST STRUTTURA PROGETTO")
print("="*60)
print()

# Test 1: Verifica file esistenti
print("1. Verifica file del progetto...")
required_files = [
    'dots_and_boxes.py',
    'mlp_evaluator.py',
    'minmax.py',
    'train_loop.py',
    'adaptive_strategy.py',
    'experiment.ipynb',
    'README.md',
    'requirements.txt'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - MANCANTE")
        missing_files.append(file)

if missing_files:
    print(f"\n   Errore: {len(missing_files)} file mancanti")
    sys.exit(1)
else:
    print(f"\n   ✓ Tutti i {len(required_files)} file presenti!")

# Test 2: Verifica sintassi Python
print("\n2. Verifica sintassi Python...")
python_files = [
    'dots_and_boxes.py',
    'mlp_evaluator.py',
    'minmax.py',
    'train_loop.py',
    'adaptive_strategy.py',
    'test_all.py'
]

syntax_errors = []
for file in python_files:
    try:
        with open(file, 'r') as f:
            code = f.read()
            compile(code, file, 'exec')
        print(f"   ✓ {file} - sintassi OK")
    except SyntaxError as e:
        print(f"   ✗ {file} - ERRORE: {e}")
        syntax_errors.append(file)

if syntax_errors:
    print(f"\n   Errore: {len(syntax_errors)} file con errori di sintassi")
    sys.exit(1)
else:
    print(f"\n   ✓ Sintassi corretta in tutti i file!")

# Test 3: Conta righe di codice
print("\n3. Statistiche codice...")
total_lines = 0
for file in python_files:
    with open(file, 'r') as f:
        lines = len(f.readlines())
        total_lines += lines
        print(f"   {file}: {lines} righe")

print(f"\n   Totale: {total_lines} righe di codice Python")

# Test 4: Verifica documentazione
print("\n4. Verifica documentazione...")
doc_files = ['README.md', 'ISTRUZIONI.md', 'requirements.txt']
for file in doc_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"   ✓ {file} ({size} bytes)")
    else:
        print(f"   ⚠ {file} - opzionale, non presente")

print("\n" + "="*60)
print("RISULTATO: Struttura progetto VALIDA ✓")
print("="*60)
print()
print("NOTA: Per eseguire il codice servono le dipendenze:")
print("  - numpy")
print("  - torch (PyTorch)")
print("  - matplotlib, seaborn, pandas")
print("  - jupyter")
print()
print("PROSSIMO PASSO: Installare pip e le dipendenze")
print("Vedi ISTRUZIONI.md per i dettagli")
print()
