import argparse,yaml,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch, numpy as np
from torch.utils.data import DataLoader
from src.data import PhoneDataset
from src.model import SegFormer5
from src.metrics import confusion,scores,boundary_f1,critical_boundary_f1
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',choices=['train','val','test'],default='test'); a=ap.parse_args()
c=yaml.safe_load(open(a.config)); sp=c[f'{a.split}_split']; ds=PhoneDataset(c['images_dir'],c['masks_dir'],sp,c['img_size'],False); dl=DataLoader(ds,batch_size=c['batch_size'],shuffle=False,num_workers=c['num_workers']); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=SegFormer5(c['model_name'],4,False).to(dev); z=torch.load(a.checkpoint,map_location=dev); m.load_state_dict(z['model']); m.eval(); cm=np.zeros((4,4),np.int64); bfs=[]; cbfs=[]; violations=[]; editable=[]
with torch.no_grad():
 for x,y,_ in dl:
  p=torch.nn.functional.interpolate(m(x.to(dev)),size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1).cpu().numpy(); yy=y.numpy()
  for pp,t in zip(p,yy):
   cm+=confusion(pp,t,4); bfs.append(boundary_f1(pp,t)); cbfs.append(critical_boundary_f1(pp,t)); editable.append((pp==1).astype(np.uint8)); violations.append(((t==2)|(t==3)) & (pp==1))
print('Class: IoU / F1')
for i,(iou,f1) in enumerate(scores(cm)): print(i,round(iou,5),round(f1,5))
print('Mean IoU:',round(np.mean([x[0] for x in scores(cm)]),5))
print('Boundary F1 (any object edge):',round(float(np.mean(bfs)),5))
print('Critical boundary F1 (cover vs camera/frame):',round(float(np.mean(cbfs)),5))
print('Protected false-positive rate:',round(float(np.sum(violations)/max(1,sum(v.size for v in violations))),8))
