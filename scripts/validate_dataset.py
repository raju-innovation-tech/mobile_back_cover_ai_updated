import argparse,sys
from pathlib import Path
from PIL import Image
import numpy as np
EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
ap=argparse.ArgumentParser(); ap.add_argument('--images',required=True); ap.add_argument('--masks',required=True); a=ap.parse_args()
ims={p.stem:p for p in Path(a.images).iterdir() if p.suffix.lower() in EXT}; masks={p.stem:p for p in Path(a.masks).glob('*.png')}
missing=sorted(set(ims)-set(masks)); extra=sorted(set(masks)-set(ims)); bad=[]
for s,p in ims.items():
    if s not in masks: continue
    try:
        with Image.open(p) as im, Image.open(masks[s]) as ma:
            if im.size!=ma.size: bad.append((s,'size',im.size,ma.size)); continue
            x=np.asarray(ma.convert('L'))
            vals=set(np.unique(x).tolist())
            if not vals.issubset({0,1,2,3}): bad.append((s,'mask_values',sorted(vals)))
            if (x==1).sum()==0: bad.append((s,'no_class_1'))
    except Exception as e: bad.append((s,'read_error',str(e)))
print(f'Images: {len(ims)} | Masks: {len(masks)} | Missing masks: {len(missing)} | Extra masks: {len(extra)} | Bad: {len(bad)}')
for x in missing[:20]: print('MISSING',x)
for x in bad[:50]: print('BAD',x)
sys.exit(1 if missing or bad else 0)
