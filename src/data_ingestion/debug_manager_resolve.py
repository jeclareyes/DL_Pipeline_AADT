import sys
from pathlib import Path
sys.path.insert(0,'src')
from data_ingestion.data_processing import DataManager

for name in ['Linköping','Linkping','Linkoping','Linkoping'.encode('utf-8').decode('latin1')]:
    try:
        m = DataManager(network_name=name)
        print('network_name:', repr(name))
        print(' resolved network_path ->', m.network_path)
        print(' resolved flow_path    ->', m.flow_path)
        print(' resolved od_path      ->', m.od_path)
        print(' resolved node_path    ->', m.node_path)
        print('exists od_path:', Path(m.od_path).exists())
    except Exception as e:
        print('Error creating DataManager for', name, ':', e)

print('\nFolders under data/interim:')
base = Path('data') / 'interim'
if base.exists():
    for p in base.iterdir():
        print(' -', p.name)
else:
    print(' data/interim does not exist')

