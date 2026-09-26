# Accuracy update — what changed and why

**Update:** the pipeline now uses the 4-class scheme from the actual
annotation spec (0=background, 1=cover — including logo/text/sticker/
ring/design, 2=camera, 3=frame). An earlier pass had a 5th class splitting
logo/design out separately; that's been removed everywhere (model output
head, loss, metrics, editable-mask logic in infer.py, dataset validation,
docs). If you already annotated any masks with a value of `4` in them,
re-map those pixels to `1` before training — `validate_dataset.py` will
now reject masks containing a `4`.

Goal: push boundary precision on the editable-cover vs protected
(camera/frame) boundary as close to pixel-perfect as possible, since that
boundary is the entire point of this pipeline.

## Fixed
- **`scripts/infer.py`**: `train.yaml` declared `inference.multiscale:
  [0.75, 1.0, 1.25]` but the old `predict()` only ever ran flip-TTA at a
  single scale — the multiscale list was dead config. Now implemented for
  real: each scale re-letterboxes the full image, runs the model, resamples
  probabilities to source resolution, and averages.

## Added
- **High-res camera-module refinement pass** (`hires_camera_refine` in
  `infer.py`): the coarse pass decides everything at `img_size` (e.g. 768),
  which throws away boundary detail on large source photos. A padded crop
  around the coarse camera prediction is now re-run at a higher resolution
  and spliced back in.
- **Guided-filter boundary snap** (`guided_filter` in `infer.py`): refines
  the soft editable-probability map against the source image's own edges
  (box-filter implementation of He et al.'s guided filter, no extra
  dependency) so the boundary aligns with the real bezel/frame line rather
  than the network's raw prediction.
- **Asymmetric protected-leakage penalty** in `src/losses.py`
  (`ComboLoss.leak`): predicting an editable class on a truly-protected
  pixel is now penalized harder than the reverse, directly targeting the
  "protected false-positive rate near zero" requirement.
- **Per-class-weighted Dice** in `src/losses.py` (was unweighted, so it
  worked against the class-weighting intent elsewhere in the loss).
- **Boundary-loss upweighting near protected edges** — the existing
  boundary term now weights camera/frame-adjacent edges 3x higher than
  other object edges.
- **Critical boundary F1** (`src/metrics.py: critical_boundary_f1`,
  wired into `scripts/evaluate.py`) — measures boundary F1 restricted to
  the editable<->protected transition specifically, instead of the
  generic any-object-vs-background boundary the old metric measured.
- **Protected-leak-aware checkpoint selection** in `scripts/train.py` —
  best-checkpoint scoring now includes the validation protected-leak rate,
  not just editable/mean IoU.
- **Training augmentation**: added scale/rotation jitter (`A.Affine`) so
  the model actually learns scale invariance — this is what makes the new
  multi-scale TTA a real ensemble instead of extrapolation to resolutions
  never seen in training — plus shadow/glare, motion blur, and small
  occlusion (`CoarseDropout`) augmentation for real handheld-photo
  robustness.

## Config
`configs/train.yaml` gained: `loss.focal_weight`, `loss.leak_weight`,
`inference.guided_filter*`, `inference.hires_refine*`. Class weights for
camera/frame raised from 2.0 to 2.5.

## Still worth doing if you want to push further
- **Retrain from scratch** with the updated loss/augmentation — these code
  changes don't retroactively improve an existing checkpoint; `outputs/best.pt`
  needs to be regenerated.
- **Dataset size/diversity** is still the dominant lever. The README
  recommends 1000 images as a strong first set; for genuinely
  best-in-class accuracy on unusual camera-island shapes (periscope zoom,
  pill-shaped islands, glass-back sapphire frames) you want deliberate
  hard-example coverage of those, not just volume.
- **Encoder ensembling** (e.g. a second `mit-b4` model with a different
  seed, averaged with the `mit-b5` at inference) is a standard next step
  once the above is retrained and evaluated, if you still need more.
- Albumentations' `CoarseDropout`/`RandomShadow` keyword arguments have
  changed across versions — verify against your installed
  `albumentations` version (`pip show albumentations`) before the first
  training run; a version mismatch there will raise a clear `TypeError`
  at augmentation-build time, not silently misbehave.
