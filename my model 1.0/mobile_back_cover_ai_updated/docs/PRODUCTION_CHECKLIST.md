# Production checklist

- [ ] Every image has exactly one same-stem mask PNG.
- [ ] Image and mask dimensions are identical.
- [ ] Mask values are only 0..3.
- [ ] Whole camera box/module is Class 2.
- [ ] Visible sidewalls/frame/buttons/ports are Class 3.
- [ ] Logo/text/sticker/ring/design that should be recolored is Class 1 (same class as the plain cover — there is no separate logo/design class).
- [ ] No model-family leakage across splits.
- [ ] Test set is untouched during training.
- [ ] Protected-region false-positive rate is near zero.
- [ ] Boundary F1 is high.
- [ ] Critical boundary F1 (cover vs camera/frame specifically, from `evaluate.py`) is high — generic boundary F1 alone can look good while this specific edge is still soft.
- [ ] Checkpoint was selected with the protected-leak-aware score in `train.py` (not editable/mean IoU alone).
- [ ] Pixel QA reports zero protected pixels changed.
- [ ] If dataset has >~1500 pairs and a GPU with headroom, consider training two encoder sizes (e.g. mit-b5 + a second seed or `mit-b4`) and ensembling their softmax outputs at inference for a further accuracy bump.
