import argparse, yaml, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, cv2, torch
import torch.nn.functional as TF
from PIL import Image
from src.model import SegFormer5
from src.colorize import recolor_rgb
from src.common import save_mask
from src.data import MEAN, STD

# ---------------------------------------------------------------------------
# Core prep: aspect-preserving letterbox into a `size` x `size` canvas.
# ---------------------------------------------------------------------------
def prep(img, size):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    c = np.zeros((size, size, 3), np.uint8)
    x = (size - nw) // 2; y = (size - nh) // 2
    c[y:y + nh, x:x + nw] = r
    a = c.astype(np.float32) / 255.
    a = (a - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float(), (x, y, nw, nh)


def _round32(v, lo=64):
    return max(lo, int(round(v / 32.0)) * 32)


def predict(model, img, base_size, device, tta=True, scales=(1.0,)):
    """
    Multi-scale + horizontal-flip TTA that actually uses the `multiscale`
    list from train.yaml (previously declared in config but never applied --
    only flip-TTA ran). Each scale re-letterboxes the *full* image into a
    differently-sized square canvas (same "fit as large as possible"
    convention used in training), runs the model, then resamples the
    per-class probabilities back to the original image resolution before
    averaging. Averaging in probability space at native resolution (rather
    than averaging low-res argmax masks) is what actually buys boundary
    precision from multi-scale TTA.
    """
    oh, ow = img.shape[:2]
    acc = None
    n = 0
    for s in scales:
        size = _round32(base_size * s)
        x, meta = prep(img, size)
        x = x[None].to(device)
        variants = [x]
        if tta:
            variants.append(torch.flip(x, [-1]))
        for v in variants:
            with torch.no_grad():
                q = torch.softmax(model(v), 1)
                q = TF.interpolate(q, size=(size, size), mode='bilinear', align_corners=False)
            if v is not x:
                q = torch.flip(q, [-1])
            xx, yy, nw, nh = meta
            crop = q[:, :, yy:yy + nh, xx:xx + nw]
            crop = TF.interpolate(crop, size=(oh, ow), mode='bilinear', align_corners=False)
            acc = crop if acc is None else acc + crop
            n += 1
    probs = (acc / n)[0].cpu().numpy()  # (C, H, W) at original resolution
    pred = probs.argmax(0).astype(np.uint8)
    return pred, probs


def guided_filter(guide_gray, src, radius=8, eps=1e-3):
    """
    Fast guided filter (He et al.) using only cv2.boxFilter, no ximgproc
    dependency. Used to snap a soft editable-probability map onto the
    ACTUAL image edges (camera bezel, frame line) instead of trusting the
    network's low-resolution boundary verbatim. This is the same technique
    used for alpha-matte refinement, applied here to the editable/protected
    boundary specifically.
    """
    guide = guide_gray.astype(np.float32) / 255.0
    p = src.astype(np.float32)
    r = radius
    mean_I = cv2.boxFilter(guide, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    corr_I = cv2.boxFilter(guide * guide, -1, (r, r))
    corr_Ip = cv2.boxFilter(guide * p, -1, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def hires_camera_refine(model, img, probs, base_size, device, tta, pad_ratio, refine_size):
    """
    The coarse pass runs everything through a `base_size` square canvas, so
    a 3000x4000 source photo has its camera-module boundary decided at
    ~768px equivalent resolution -- this is the single biggest source of
    boundary error for high-res source photos. Here we crop a padded box
    around the coarse camera-module (class 2) prediction, re-run the same
    model on that crop upsampled to `refine_size`, and splice the refined
    probabilities back in. Only the crop region is touched; everything
    outside it keeps the coarse-pass result.
    """
    cam = (probs.argmax(0) == 2).astype(np.uint8)
    if cam.sum() < 25:
        return probs
    ys, xs = np.where(cam > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = cam.shape
    ph = int((y1 - y0 + 1) * pad_ratio) + 8
    pw = int((x1 - x0 + 1) * pad_ratio) + 8
    cy0, cy1 = max(0, y0 - ph), min(h, y1 + 1 + ph)
    cx0, cx1 = max(0, x0 - pw), min(w, x1 + 1 + pw)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return probs
    cpred, cprobs = predict(model, crop, refine_size, device, tta=tta, scales=(1.0,))
    probs = probs.copy()
    probs[:, cy0:cy1, cx0:cx1] = cprobs
    return probs


def conservative(mask, min_area_ratio=.00002, boundary_erode=1):
    editable = (mask == 1).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(editable, 8)
    out = np.zeros_like(editable)
    minarea = max(16, int(editable.size * min_area_ratio))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= minarea:
            out[lab == i] = 1
    if boundary_erode > 0:
        k = np.ones((3, 3), np.uint8)
        out = cv2.erode(out, k, iterations=boundary_erode)
    return out.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--hex', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--mask_output')
    ap.add_argument('--no-tta', action='store_true')
    ap.add_argument('--no-multiscale', action='store_true')
    ap.add_argument('--no-hires-refine', action='store_true')
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config))
    inf = c.get('inference', {})
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = SegFormer5(c['model_name'], 4, False).to(dev)
    z = torch.load(a.checkpoint, map_location=dev)
    m.load_state_dict(z['model']); m.eval()

    img = np.asarray(Image.open(a.image).convert('RGB'))
    tta = not a.no_tta
    scales = tuple(inf.get('multiscale', [1.0])) if not a.no_multiscale else (1.0,)

    pred, probs = predict(m, img, c['img_size'], dev, tta=tta, scales=scales)

    if inf.get('hires_refine', True) and not a.no_hires_refine:
        probs = hires_camera_refine(
            m, img, probs, c['img_size'], dev, tta,
            inf.get('hires_pad_ratio', 0.12), inf.get('hires_refine_size', 1024))

    if inf.get('guided_filter', True):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        editable_soft = probs[1]
        editable_soft = guided_filter(gray, editable_soft, inf.get('guided_filter_radius', 8), inf.get('guided_filter_eps', 1e-3))
        raw = probs.argmax(0).astype(np.uint8)
        # Guided filter only refines the editable/protected boundary; it
        # never overrides a hardware class with an editable one outright --
        # it's applied on top of, then intersected with, the conservative
        # connected-component mask below for safety.
        edit_soft_mask = editable_soft >= inf.get('threshold', 0.5)
    else:
        raw = probs.argmax(0).astype(np.uint8)
        edit_soft_mask = (raw == 1)

    hard_editable = (raw == 1)
    combined = (edit_soft_mask & hard_editable).astype(np.uint8) * 1
    combined_full = raw.copy()
    combined_full[(combined == 0) & hard_editable] = 3  # fall back to protected-safe when guided filter disagrees

    edit = conservative(combined_full, inf.get('min_component_area_ratio', 0.00002), inf.get('boundary_erode_px', 1))

    out = recolor_rgb(img, edit, a.hex)
    Image.fromarray(out).save(a.output, quality=95 if Path(a.output).suffix.lower() in ['.jpg', '.jpeg'] else None)
    mo = a.mask_output or str(Path(a.output).with_name(Path(a.output).stem + '_mask.png'))
    save_mask((edit * 255).astype(np.uint8), mo)
    print('Output:', a.output)
    print('Mask:', mo)
    print('Scales used:', scales, '| TTA flip:', tta, '| hires camera refine:', inf.get('hires_refine', True) and not a.no_hires_refine)
    print('Original size:', img.shape[1], img.shape[0])


if __name__ == '__main__':
    main()
