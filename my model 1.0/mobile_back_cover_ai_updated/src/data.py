from pathlib import Path
import numpy as np, torch, cv2
from torch.utils.data import Dataset
import albumentations as A
from src.common import decode_class_mask, find_mask_path

MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

def letterbox(img,mask,size):
    h,w=img.shape[:2]; s=min(size/w,size/h); nw,nh=max(1,round(w*s)),max(1,round(h*s))
    im=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
    ma=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST)
    canvas=np.zeros((size,size,3),np.uint8); mcanvas=np.zeros((size,size),np.uint8)
    x=(size-nw)//2; y=(size-nh)//2; canvas[y:y+nh,x:x+nw]=im; mcanvas[y:y+nh,x:x+nw]=ma
    return canvas,mcanvas,(x,y,nw,nh)

def build_aug(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        # Scale/rotation jitter so the model actually sees the phone at
        # varying apparent scale during training -- this is what makes the
        # multi-scale TTA in infer.py (config: inference.multiscale) a real
        # ensemble instead of blind extrapolation to unseen resolutions.
        # border_mode constant + mask_value 0 keeps padded regions as
        # background (class 0), consistent with the letterbox convention.
        A.Affine(scale=(0.82, 1.18), rotate=(-6, 6), shear=(-2, 2),
                  mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(0.15,0.15),A.ColorJitter(0.12,0.12,0.12,0.05)],p=0.35),
        # Real phone photos: glare off glossy backs, shadows from the hand
        # holding the phone, and blur from a handheld shot.
        A.OneOf([A.RandomShadow(p=1.0), A.RandomSunFlare(p=1.0)], p=0.15),
        A.MotionBlur(blur_limit=5,p=0.1),
        # Small occlusions (fingers, cables) so the model doesn't collapse
        # on partial views of the cover/camera/frame. NOTE: param names for
        # CoarseDropout changed between albumentations versions -- confirm
        # against `pip show albumentations` and adjust (num_holes_range vs
        # min_holes/max_holes, fill vs mask_fill_value) if this errors.
        A.CoarseDropout(num_holes_range=(1,3),hole_height_range=(0.02,0.08),hole_width_range=(0.02,0.08),fill=0,fill_mask=0,p=0.15),
        A.GaussNoise(std_range=(0.01,0.04),p=0.15),
        A.ImageCompression(quality_range=(80,100),p=0.15),
    ])

class PhoneDataset(Dataset):
    def __init__(self,images_dir,masks_dir,split_file,size=768,train=False):
        self.images_dir=Path(images_dir); self.masks_dir=Path(masks_dir); self.size=size; self.train=train
        self.stems=[x.strip() for x in Path(split_file).read_text().splitlines() if x.strip()]
        self.aug=build_aug(size) if train else None
    def __len__(self): return len(self.stems)
    def __getitem__(self,i):
        stem=self.stems[i]
        ip=next(self.images_dir.glob(stem+'.*'))
        mp=find_mask_path(self.masks_dir, stem)
        if mp is None:
            raise FileNotFoundError(f'No mask found for stem "{stem}" in {self.masks_dir} (looked for {stem}.png and {stem}_mask.png)')
        img=cv2.cvtColor(cv2.imread(str(ip),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        mask=decode_class_mask(mp)
        img,mask,_=letterbox(img,mask,self.size)
        if self.aug:
            z=self.aug(image=img,mask=mask); img,mask=z['image'],z['mask']
        img=img.astype(np.float32)/255.; img=(img-MEAN)/STD
        return torch.from_numpy(img.transpose(2,0,1)).float(),torch.from_numpy(mask.astype(np.int64)),stem
