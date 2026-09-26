# Mobile Back Cover AI — Complete Training & Inference Pipeline

This project is designed for one job: recolor ONLY the smartphone rear-cover surface (including its logo/text/sticker/ring/design, which are NOT separately classed) while preserving camera hardware, frame/sidewalls, buttons and ports.

## Final mask contract

Use a single-channel PNG with integer pixel values:

- `0` Background — protected
- `1` Rear-cover surface, INCLUDING logo / mobile brand text / sticker / ring / printed design — recolor
- `2` COMPLETE camera box/module + lenses + flash + sensors — protected
- `3` Frame / sidewalls / buttons / ports — protected

There is no separate class for logo/text/design — they are part of Class 1
and get recolored along with the rest of the cover.

Final editable mask is exactly: `mask == 1`.

**Camera rule:** the entire visible camera box/module is Class 2. Do not paint only the lenses as Class 2.

## Dataset

Put paired files here:

```
data/images/phone_0001.jpg

data/masks/phone_0001.png
```

The stem must match. Every mask must have EXACTLY the same width and height as its image. Masks must be single-channel PNGs containing only values 0..3.

The supplied starter images can be removed; they are only placeholders.

## Recommended dataset size

1000 real, diverse images is a strong first production dataset. More diverse images are better than near-duplicates. Keep phone-model families separated between train/validation/test where possible to measure real generalization.

## Install

Python 3.10 or 3.11 recommended.

1. Install a PyTorch build appropriate for your GPU from the official PyTorch instructions.
2. Then:

```
pip install -r requirements.txt
```

3. Check:

```
python scripts/check_environment.py
```

## Validate dataset

```
python scripts/validate_dataset.py --images data/images --masks data/masks
```

This checks matching pairs, dimensions, mask values, unreadable files and suspicious empty masks.

## Create splits

For ordinary filenames:

```
python scripts/make_splits.py --images data/images --masks data/masks --out data/splits --val 0.10 --test 0.10 --seed 42
```

For stronger leakage control, provide a CSV with `stem,group` where `group` is the phone/model family:

```
python scripts/make_splits.py --images data/images --masks data/masks --out data/splits --groups data/groups.csv --val 0.10 --test 0.10
```

## Train

Edit `configs/train.yaml` if needed, then:

```
python scripts/train.py --config configs/train.yaml
```

The trainer uses a pretrained SegFormer encoder, class-balanced cross entropy + Dice + boundary-aware loss, strong but geometry-safe augmentation, mixed precision, gradient clipping, cosine schedule, checkpoint/resume, validation metrics and best-checkpoint selection.

For a GPU with more memory, increase batch size. For lower memory, reduce `img_size` or batch size.

## Evaluate

```
python scripts/evaluate.py --config configs/train.yaml --checkpoint outputs/best.pt --split test
```

Metrics include per-class IoU/F1, editable-mask IoU, protected-region false-positive rate, protected-pixel violations, boundary F1 and exact mask dimensions.

## Inference / recoloring

```
python scripts/infer.py --config configs/train.yaml --checkpoint outputs/best.pt --image path/to/phone.jpg --hex '#00A651' --output outputs/green.png
```

Inference now runs, controlled by `configs/train.yaml -> inference:`:

- **Real multi-scale TTA** (`multiscale: [0.75, 1.0, 1.25]`) — the image is re-letterboxed at each scale, run through the model, and the resulting per-class probabilities are resampled to the source resolution and averaged together with horizontal-flip TTA. (Previously this list was declared in the config but silently ignored — only flip TTA ran.)
- **High-resolution camera-module refinement** (`hires_refine: true`) — after the coarse full-image pass, a padded crop around the predicted camera module is re-run at `hires_refine_size` (default 1024) and spliced back in, so the camera boundary isn't decided at whatever resolution a 3000x4000 source photo got downsampled to for the coarse pass.
- **Guided-filter boundary snap** (`guided_filter: true`) — the soft editable-probability map is refined against the source image's own luminance edges (same technique used for alpha-matte refinement), so the predicted boundary snaps onto the real camera bezel / frame line instead of trusting the network's raw (lower-resolution) edge.

Disable any of these per-run with `--no-multiscale`, `--no-hires-refine`, `--no-tta`.

Batch:

```
python scripts/batch_infer.py --config configs/train.yaml --checkpoint outputs/best.pt --input_dir data/images --output_dir outputs/recolored --hex '#00A651'
```

The compositor is deterministic: outside the final editable mask, source pixels are copied exactly. Inside the editable mask, the target color is applied while preserving the source luminance/texture using LAB-space blending.

## Pixel QA

For a single result:

```
python scripts/pixel_qa.py --original path/to/phone.jpg --result outputs/green.png --mask outputs/mask.png
```

The QA checks that every pixel outside the editable mask is byte-for-byte unchanged (after loading both images into the same RGB representation).

## Important annotation rules

1. Follow the real physical boundary of the rear cover.
2. Whole camera box/module = Class 2.
3. Visible frame/sidewalls/buttons/ports = Class 3.
4. Logo/brand text/sticker/ring/design = Class 1 (same as plain cover) and WILL be recolored.
5. If uncertain whether a pixel is cover or protected hardware, mark it protected.
6. Do not resize masks to make them fit. Fix the source/mask pair instead.
7. Avoid near-duplicate leakage across train/validation/test.

## What you provide

You only need to provide the paired image + mask dataset. After the dataset is placed into `data/images` and `data/masks`, the included scripts handle validation, splitting, training, evaluation, inference and QA.

## Production note

No generative image model is used to redraw the phone. The model predicts a mask; the original image is then composited deterministically. This is intentional because the requirement is pixel preservation, not image generation.
