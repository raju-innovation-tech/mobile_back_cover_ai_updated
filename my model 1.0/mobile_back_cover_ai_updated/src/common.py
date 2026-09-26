from pathlib import Path
import random, numpy as np
from PIL import Image

VALID_EXT={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

def seed_everything(seed=42):
    # torch is imported lazily here (not at module level) so that
    # lightweight, torch-free scripts like validate_dataset.py and
    # dataset_report.py can use decode_class_mask()/load_mask() from this
    # file without requiring torch to be installed just to check masks.
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def list_images(d):
    return sorted([p for p in Path(d).iterdir() if p.suffix.lower() in VALID_EXT])

def load_rgb(path): return Image.open(path).convert('RGB')

# The annotation spec (and generate_exact_mask_hq.py) produce masks as
# RGB color-coded PNGs, NOT single-channel class-index PNGs:
#   black (0,0,0)     -> 0  background
#   white (255,255,255) -> 1  cover (includes logo/text/sticker/ring/design)
#   red   (255,0,0)   -> 2  camera module
#   blue  (0,0,255)   -> 3  frame/hardware
# Loading these with a plain grayscale/luminance conversion (cv2.imread
# with IMREAD_GRAYSCALE, or PIL .convert('L')) does NOT recover 0..3 --
# it produces {0, 29, 76, 255} instead, which then gets fed straight into
# cross_entropy as class labels. This is the single decode function used
# everywhere a mask is read, so that bug can't reappear in some other file.
MASK_PALETTE = {
    (0, 0, 0): 0,
    (255, 255, 255): 1,
    (255, 0, 0): 2,
    (0, 0, 255): 3,
}

def decode_class_mask(path):
    """
    Returns a single-channel uint8 array of class indices (0..3) for a
    mask file, handling both formats so old and new masks work:
      - RGB color-coded masks (the real format the annotation tooling
        produces) -- decoded via EXACT match against the 4 spec colors
        (the generator writes solid fills with no blending/anti-aliasing,
        so exact match covers effectively all pixels).
      - Already-indexed single-channel masks with raw values 0..3, kept
        for backward compatibility.
    Any pixel that doesn't exactly match one of the 4 spec colors and
    isn't already a valid index defaults to background (0) rather than
    crashing -- use `find_unexpected_mask_colors` at dataset-validation
    time to catch that case explicitly instead of silently defaulting it.
    """
    im = Image.open(path)
    if im.mode in ('L', 'P', '1'):
        arr = np.asarray(im.convert('L'), dtype=np.uint8)
        if set(np.unique(arr).tolist()).issubset({0, 1, 2, 3}):
            return arr
    arr = np.asarray(im.convert('RGB'), dtype=np.uint8)
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for (r, g, b), cls in MASK_PALETTE.items():
        if cls == 0:
            continue
        out[(arr[..., 0] == r) & (arr[..., 1] == g) & (arr[..., 2] == b)] = cls
    return out

def find_unexpected_mask_colors(path, max_report=8):
    """RGB colors present in a mask that aren't one of the 4 spec colors
    and aren't already a valid 0..3 index mask. Used by
    validate_dataset.py to flag bad annotation exports instead of letting
    decode_class_mask silently default them to background."""
    im = Image.open(path)
    if im.mode in ('L', 'P', '1'):
        arr = np.asarray(im.convert('L'), dtype=np.uint8)
        if set(np.unique(arr).tolist()).issubset({0, 1, 2, 3}):
            return []
    arr = np.asarray(im.convert('RGB'), dtype=np.uint8).reshape(-1, 3)
    colors = set(map(tuple, np.unique(arr, axis=0).tolist()))
    bad = sorted(c for c in colors if c not in MASK_PALETTE)
    return bad[:max_report]

def find_mask_path(masks_dir, stem):
    """
    Locates the mask file for an image stem, accepting either naming
    convention in use across this project: `<stem>.png` (the convention
    documented in the README) or `<stem>_mask.png` (what
    generate_exact_mask_hq.py actually writes by default). Returns None
    if neither exists.
    """
    masks_dir = Path(masks_dir)
    for cand in (masks_dir / f'{stem}.png', masks_dir / f'{stem}_mask.png'):
        if cand.exists():
            return cand
    return None

def load_mask(path): return decode_class_mask(path)

def save_mask(arr,path):
    Image.fromarray(arr.astype(np.uint8),'L').save(path)

