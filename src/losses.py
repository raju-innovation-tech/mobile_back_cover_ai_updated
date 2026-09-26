import torch, torch.nn as nn, torch.nn.functional as F

class ComboLoss(nn.Module):
    """
    ce + dice + focal + boundary + protected-leakage penalty.

    protected classes = {2 (camera), 3 (frame/hardware)}
    editable classes   = {1 (cover -- includes logo/text/sticker/ring/design)}

    The extra `leak` term is asymmetric on purpose: predicting an editable
    class on a pixel whose ground truth is protected hardware is penalized
    harder than the reverse (missing a sliver of cover is recoverable;
    recoloring part of a camera lens is not). This directly targets the
    production metric in evaluate.py / PRODUCTION_CHECKLIST.md
    ("protected-region false-positive rate is near zero").
    """
    def __init__(self, weights=None, ce=1., dice=1., boundary=.25, gamma=2.,
                 focal_w=0.25, leak=2.0, protected=(2, 3), editable=(1,)):
        super().__init__()
        self.register_buffer('w', torch.tensor(weights or [1, 1, 1, 1], dtype=torch.float))
        self.ce = ce; self.dice = dice; self.boundary = boundary; self.gamma = gamma
        self.focal_w = focal_w; self.leak = leak
        self.protected = protected; self.editable = editable

    def forward(self, logits, target):
        w = self.w.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=w)

        p = logits.softmax(1)
        oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()

        # Per-class weighted Dice (was unweighted before -> camera/frame classes
        # were getting the same say as background despite class_weights implying
        # they matter more).
        dims = (0, 2, 3)
        inter = (p * oh).sum(dims); den = (p + oh).sum(dims)
        dice_per_class = 1 - ((2 * inter + 1) / (den + 1))
        dl = (dice_per_class * w).sum() / w.sum()

        focal = ((1 - p.clamp_min(1e-6)).pow(self.gamma) * (-torch.log(p.clamp_min(1e-6))) * oh).sum() / (oh.sum() + 1)

        # Boundary term, upweighted specifically at protected<->editable transitions
        # (camera/frame edges), since that boundary is the one that must never leak.
        tgt_f = target[:, None].float()
        edge = ((F.max_pool2d(tgt_f, 3, 1, 1) - F.avg_pool2d(tgt_f, 3, 1, 1)).abs() > 0.01).float()[:, 0]
        is_protected = torch.zeros_like(tgt_f[:, 0])
        for c in self.protected:
            is_protected = is_protected + (target == c).float()
        crit_edge = edge * (1.0 + 2.0 * (is_protected > 0).float())  # 3x weight near hardware edges
        bce = F.cross_entropy(logits, target, reduction='none')
        bl = (bce * crit_edge).sum() / (crit_edge.sum() + 1)

        # Asymmetric protected-leakage penalty: probability mass placed on
        # editable classes at truly-protected pixels.
        protected_mask = torch.zeros_like(tgt_f[:, 0])
        for c in self.protected:
            protected_mask = protected_mask + (target == c).float()
        editable_prob = sum(p[:, c] for c in self.editable)
        leak_term = (editable_prob * protected_mask).sum() / (protected_mask.sum() + 1)

        return (self.ce * ce + self.dice * dl + self.focal_w * focal
                + self.boundary * bl + self.leak * leak_term)
