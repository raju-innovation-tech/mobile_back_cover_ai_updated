import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument('--images',required=True);ap.add_argument('--masks',required=True);a=ap.parse_args(); ims=sorted(Path(a.images).iterdir()); rows=[]
for p in ims:
 if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}: continue
 m=Path(a.masks)/(p.stem+'.png')
 if not m.exists(): continue
 with Image.open(p) as im,Image.open(m) as ma:
  x=np.asarray(ma.convert('L')); rows.append((p.stem,im.size,[(x==i).sum() for i in range(4)]))
print('pairs:',len(rows));
for r in rows[:20]: print(r)
