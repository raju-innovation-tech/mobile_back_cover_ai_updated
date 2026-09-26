import argparse,subprocess
from pathlib import Path
EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--input_dir',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--hex',required=True); a=ap.parse_args(); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
for p in sorted(Path(a.input_dir).iterdir()):
 if p.suffix.lower() not in EXT: continue
 o=out/(p.stem+'.png'); subprocess.run(['python','scripts/infer.py','--config',a.config,'--checkpoint',a.checkpoint,'--image',str(p),'--hex',a.hex,'--output',str(o)],check=True)
