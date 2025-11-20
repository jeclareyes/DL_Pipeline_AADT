import sys
from pathlib import Path
sys.path.insert(0, 'src')

from data_ingestion.data_processing import DataManager
from data_ingestion.processing_modules.od_matrix_generator import MultidayODMatrixGenerator, ODMatrixGenerator

def main():
    print('Creating DataManager for Linköping...')
    manager = DataManager(network_name='Linköping')
    print('Resolved paths:')
    print(' network_path ->', manager.network_path)
    print(' flow_path    ->', manager.flow_path)
    print(' od_path      ->', manager.od_path)
    print(' node_path    ->', manager.node_path)

    od_path = Path(manager.od_path)
    print('\nCheck file existence:')
    print(' exists:', od_path.exists())
    if not od_path.exists():
        print('File does not exist; aborting inspection')
        return

    print('\nFirst 200 lines of file:')
    try:
        with open(od_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= 200:
                    break
                print(f'{i+1:04d}: {line.rstrip()}')
    except Exception as e:
        print('Error reading file:', e)

    print('\nAttempting to load using MultidayODMatrixGenerator...')
    try:
        gen = MultidayODMatrixGenerator(od_path)
        od_df, meta = gen.load()
        print('\nLoad succeeded. Metadata:')
        print(meta)
        print('\nOD dataframe head:')
        print(od_df.head())
    except Exception as e:
        print('\nLoad failed with exception:')
        import traceback
        traceback.print_exc()

    print('\nAlso try single-day OD generator as fallback (ODMatrixGenerator)')
    try:
        gen2 = ODMatrixGenerator(od_path)
        od_df2, meta2 = gen2.load()
        print('\nODMatrixGenerator load succeeded. Metadata:')
        print(meta2)
        print('\nOD dataframe head:')
        print(od_df2.head())
    except Exception as e:
        print('\nODMatrixGenerator failed as well:')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

