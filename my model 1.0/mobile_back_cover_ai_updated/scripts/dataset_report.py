import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import decode_class_mask, find_mask_path
ap=argparse.ArgumentParser();ap.add_argument('--images',required=True);ap.add_argument('--masks',required=True);a=ap.parse_args(); ims=sorted(Path(a.images).iterdir()); rows=[]
for p in ims:
 if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}: continue
 m=find_mask_path(a.masks, p.stem)
 if m is None: continue
 with Image.open(p) as im:
  x=decode_class_mask(m); rows.append((p.stem,im.size,[(x==i).sum() for i in range(4)]))
print('pairs:',len(rows));
for r in rows[:20]: print(r)
