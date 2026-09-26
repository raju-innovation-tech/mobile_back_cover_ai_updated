import argparse,random,csv,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import find_mask_path
ap=argparse.ArgumentParser(); ap.add_argument('--images',required=True); ap.add_argument('--masks',required=True); ap.add_argument('--out',required=True); ap.add_argument('--val',type=float,default=.1); ap.add_argument('--test',type=float,default=.1); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--groups'); a=ap.parse_args()
random.seed(a.seed); ims=sorted([p.stem for p in Path(a.images).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'} and find_mask_path(a.masks, p.stem) is not None])
groups={x:x for x in ims}
if a.groups:
  with open(a.groups,newline='',encoding='utf-8') as f:
    for r in csv.DictReader(f): groups[r['stem']]=r['group']
uniq=list(set(groups[x] for x in ims)); random.shuffle(uniq); n=len(uniq); nt=round(n*a.test); nv=round(n*a.val); testg=set(uniq[:nt]); valg=set(uniq[nt:nt+nv])
train=[x for x in ims if groups[x] not in testg|valg]; val=[x for x in ims if groups[x] in valg]; test=[x for x in ims if groups[x] in testg]
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for name,arr in [('train',train),('val',val),('test',test)]: (out/(name+'.txt')).write_text('\n'.join(arr)+'\n',encoding='utf-8'); print(name,len(arr))
