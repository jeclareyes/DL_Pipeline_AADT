from pathlib import Path
p = Path('data/interim/Linköping/Linköping_trips.tntp')
if not p.exists():
    p = Path('data/interim/Linköping/Linköping_trips.tntp')
print('File',p.exists(),p)
count=0
with open(p,'r',encoding='utf-8',errors='replace') as f:
    lines = f.readlines()
for i,line in enumerate(lines):
    if ':' in line:
        parts = line.split(';')
        for part in parts:
            if ':' in part:
                try:
                    dest,flow = part.split(':',1)
                    v = float(flow.strip())
                    if v>0:
                        print('LINE',i+1,lines[i-2].strip() if i-2>=0 else '', lines[i-1].strip() if i-1>=0 else '', line.strip())
                        count+=1
                        if count>30:
                            break
                except Exception:
                    pass
    if count>30:
        break
print('found',count)

