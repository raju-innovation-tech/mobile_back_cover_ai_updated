from pathlib import Path
import random, numpy as np, torch
from PIL import Image

VALID_EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def list_images(d):
    return sorted([p for p in Path(d).iterdir() if p.suffix.lower() in VALID_EXT])

def load_rgb(path): return Image.open(path).convert('RGB')
def load_mask(path): return np.asarray(Image.open(path).convert('L'), dtype=np.uint8)

def save_mask(arr,path):
    Image.fromarray(arr.astype(np.uint8),'L').save(path)
