from pathlib import Path
import sys
sys.path.insert(0,'src')
from data_ingestion.processing_modules.od_matrix_generator import MultidayODMatrixGenerator
p = Path('data/interim/Linköping/Linköping_trips.tntp')
print('path exists', p.exists(), p)
G = MultidayODMatrixGenerator(p)
lines = G._read_lines()
print('lines', len(lines))
hourly = G._parse_multiday_od_data(lines)
print('parsed hourly entries', len(hourly))
for i,entry in enumerate(hourly[:40]):
    print(i, entry)

# show sample where flow>0
nonzero = [e for e in hourly if e.get('flow',0)>0]
print('nonzero count', len(nonzero))
for i,e in enumerate(nonzero[:20]):
    print('nz', i, e)

