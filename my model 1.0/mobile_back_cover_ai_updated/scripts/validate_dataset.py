import argparse,sys
from pathlib import Path
from PIL import Image
import numpy as np
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import decode_class_mask, find_unexpected_mask_colors, find_mask_path
EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
ap=argparse.ArgumentParser(); ap.add_argument('--images',required=True); ap.add_argument('--masks',required=True); a=ap.parse_args()
ims={p.stem:p for p in Path(a.images).iterdir() if p.suffix.lower() in EXT}
masks={}
for stem in ims:
    mp=find_mask_path(a.masks, stem)
    if mp is not None: masks[stem]=mp
missing=sorted(set(ims)-set(masks))
all_mask_stems={(p.stem[:-5] if p.stem.endswith('_mask') else p.stem) for p in Path(a.masks).glob('*.png')}
extra=sorted(all_mask_stems-set(ims))
bad=[]
for s,p in ims.items():
    if s not in masks: continue
    try:
        with Image.open(p) as im, Image.open(masks[s]) as ma:
            if im.size!=ma.size: bad.append((s,'size',im.size,ma.size)); continue
            # Masks are RGB color-coded (black/white/red/blue) per the
            # annotation spec, decoded to 0..3 via the shared palette --
            # NOT a plain grayscale conversion, which would misread the
            # spec colors as {0,29,76,255}.
            stray = find_unexpected_mask_colors(masks[s])
            if stray: bad.append((s,'unexpected_mask_colors',stray))
            x=decode_class_mask(masks[s])
            vals=set(np.unique(x).tolist())
            if not vals.issubset({0,1,2,3}): bad.append((s,'mask_values',sorted(vals)))
            if (x==1).sum()==0: bad.append((s,'no_class_1'))
    except Exception as e: bad.append((s,'read_error',str(e)))
print(f'Images: {len(ims)} | Masks: {len(masks)} | Missing masks: {len(missing)} | Extra masks: {len(extra)} | Bad: {len(bad)}')
for x in missing[:20]: print('MISSING',x)
for x in bad[:50]: print('BAD',x)
sys.exit(1 if missing or bad else 0)
