import os
from pathlib import Path

# Test 1: verificar el nombre del directorio
data_dir = Path("../data/processed")
print("Contenido de data/processed:")
for item in data_dir.iterdir():
    print(f"  - {item.name} (existe: {item.exists()})")

# Test 2: probar diferentes formas de acceder
test_paths = [
    "data/processed/Linköping",
    r"data/processed/Linköping",
    "data\\processed\\Linköping",
]

print("\nProbando diferentes paths:")
for test_path in test_paths:
    p = Path(test_path)
    print(f"  {test_path} -> existe: {p.exists()}, resuelto: {p.resolve() if p.exists() else 'N/A'}")

# Test 3: listar archivos en Linköping
linkoping_dir = Path("../data/processed/Linköping")
if linkoping_dir.exists():
    print(f"\nContenido de {linkoping_dir}:")
    for item in linkoping_dir.iterdir():
        print(f"  - {item.name}")

