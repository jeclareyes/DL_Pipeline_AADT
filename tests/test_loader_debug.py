import yaml
from pathlib import Path

# Cargar config
with open('../configs/linkoping.yaml', 'r') as f:
    config = yaml.safe_load(f)

base_path_str = config['data']['base_path']
print(f"Base path string from YAML: {repr(base_path_str)}")

base_path = Path(base_path_str)
print(f"Base path object: {base_path}")
print(f"Base path exists: {base_path.exists()}")

# Intentar glob
parent = base_path.parent
pattern = base_path.name.replace('ö', '*').replace('ä', '*').replace('å', '*')
print(f"\nParent: {parent}")
print(f"Pattern: {pattern}")

matches = list(parent.glob(pattern))
print(f"Matches: {matches}")

if matches:
    actual_path = matches[0]
    print(f"\nActual path found: {actual_path}")
    print(f"Exists: {actual_path.exists()}")
    
    # Buscar archivos
    graph_file = config['data']['graph_file']
    print(f"\nGraph file from config: {repr(graph_file)}")
    
    graph_path = actual_path / graph_file
    print(f"Graph path: {graph_path}")
    print(f"Graph exists: {graph_path.exists()}")
    
    # Buscar con glob
    pattern2 = graph_file.replace('ö', '*').replace('ä', '*').replace('å', '*')
    print(f"\nGraph pattern: {pattern2}")
    matches2 = list(actual_path.glob(pattern2))
    print(f"Graph matches: {matches2}")
    
    # Buscar todos los pkl
    all_pkl = list(actual_path.glob('*.pkl'))
    print(f"\nAll PKL files: {all_pkl}")

