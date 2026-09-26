import argparse,numpy as np
from PIL import Image
ap=argparse.ArgumentParser(); ap.add_argument('--original',required=True); ap.add_argument('--result',required=True); ap.add_argument('--mask',required=True); a=ap.parse_args(); o=np.asarray(Image.open(a.original).convert('RGB')); r=np.asarray(Image.open(a.result).convert('RGB')); m=np.asarray(Image.open(a.mask).convert('L'))>0
if o.shape!=r.shape: raise SystemExit(f'SIZE MISMATCH: {o.shape} vs {r.shape}')
unchanged=np.all(o==r,axis=2); protected=~m; changed=int((~unchanged & protected).sum()); total=int(protected.sum()); print('Exact unchanged outside edit mask:',f'{1-changed/max(1,total):.8f}'); print('Protected pixels changed:',changed); print('Result size:',o.shape[1],o.shape[0]); raise SystemExit(1 if changed else 0)
