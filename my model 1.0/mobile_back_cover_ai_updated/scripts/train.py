import argparse,yaml,sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.common import seed_everything
from src.data import PhoneDataset
from src.model import SegFormer5
from src.losses import ComboLoss
from src.metrics import confusion,scores,boundary_f1

def evaluate(model,loader,device):
    model.eval(); cm=torch.zeros((4,4),dtype=torch.int64); bf=[]
    with torch.no_grad():
      for x,y,_ in loader:
        p=model(x.to(device)).argmax(1); p=torch.nn.functional.interpolate(p[:,None].float(),size=y.shape[-2:],mode='nearest')[:,0].long().cpu().numpy(); y=y.numpy()
        for pp,yy in zip(p,y): cm+=torch.from_numpy(confusion(pp,yy,4)); bf.append(boundary_f1(pp,yy))
    sc=scores(cm.numpy()); return cm.numpy(),sc,float(sum(bf)/max(1,len(bf)))

def protected_leak_rate(cm):
    # Fraction of truly-protected pixels (camera=2, frame=3) predicted as
    # editable cover (class 1, which already includes logo/text/sticker/
    # design per the 4-class annotation spec). This is the
    # production-critical metric from PRODUCTION_CHECKLIST.md ("protected
    # FP rate near zero"), so it is folded directly into checkpoint
    # selection below rather than only being reported at evaluate.py time.
    protected_total = cm[2].sum() + cm[3].sum()
    leaked = cm[2,1] + cm[3,1]
    return float(leaked) / float(protected_total + 1e-9)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args(); c=yaml.safe_load(open(a.config)); seed_everything(c['seed']); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=Path(c['output_dir']); out.mkdir(exist_ok=True)
 tr=PhoneDataset(c['images_dir'],c['masks_dir'],c['train_split'],c['img_size'],True); va=PhoneDataset(c['images_dir'],c['masks_dir'],c['val_split'],c['img_size'],False)
 tl=DataLoader(tr,batch_size=c['batch_size'],shuffle=True,num_workers=c['num_workers'],pin_memory=True); vl=DataLoader(va,batch_size=c['batch_size'],shuffle=False,num_workers=c['num_workers'],pin_memory=True)
 model=SegFormer5(c['model_name'],c['num_classes'],c['pretrained']).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['epochs']); lossfn=ComboLoss(c['class_weights'],ce=c['loss']['ce'],dice=c['loss']['dice'],boundary=c['loss']['boundary'],gamma=c['loss'].get('focal_gamma',2.0),focal_w=c['loss'].get('focal_weight',0.25),leak=c['loss'].get('leak_weight',2.0)); scaler=torch.amp.GradScaler('cuda',enabled=(c['amp'] and dev.type=='cuda'))
 best=-1; bad=0; start=0; ck=out/'last.pt'
 if ck.exists():
  z=torch.load(ck,map_location=dev); model.load_state_dict(z['model']); opt.load_state_dict(z['opt']); sch.load_state_dict(z['sch']); scaler.load_state_dict(z['scaler']); start=z['epoch']+1; best=z['best']; print('Resuming epoch',start)
 for ep in range(start,c['epochs']):
  model.train(); total=0
  bar=tqdm(tl,desc=f'epoch {ep+1}/{c["epochs"]}')
  for x,y,_ in bar:
   x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,enabled=(c['amp'] and dev.type=='cuda')): logits=model(x); logits=torch.nn.functional.interpolate(logits,size=y.shape[-2:],mode='bilinear',align_corners=False); loss=lossfn(logits,y)
   scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip']); scaler.step(opt); scaler.update(); total+=loss.item(); bar.set_postfix(loss=f'{loss.item():.4f}')
  sch.step(); cm,sc,bf=evaluate(model,vl,dev); editable_iou=(cm[1,1])/(cm[1].sum()+cm[:,1].sum()-cm[1,1]+1e-9)
  mean_iou=sum(x[0] for x in sc)/4; leak=protected_leak_rate(cm)
  # Checkpoint selection now directly penalizes protected-region leakage
  # instead of relying only on editable/mean IoU, which could stay high
  # even while a small but critical fraction of camera/frame pixels leak
  # into the editable classes.
  score=.6*editable_iou+.2*mean_iou+.2*(1.0-min(1.0,leak*50))
  print('val loss',total/max(1,len(tl)),'meanIoU',mean_iou,'editableIoU',editable_iou,'boundaryF1',bf,'protectedLeakRate',leak)
  state={'epoch':ep,'model':model.state_dict(),'opt':opt.state_dict(),'sch':sch.state_dict(),'scaler':scaler.state_dict(),'best':best,'config':c}
  torch.save(state,ck)
  if score>best: best=score; state['best']=best; torch.save(state,out/'best.pt'); bad=0
  else: bad+=1
  if bad>=c['patience']: print('Early stopping'); break
 print('Best score:',best)
if __name__=='__main__': main()
