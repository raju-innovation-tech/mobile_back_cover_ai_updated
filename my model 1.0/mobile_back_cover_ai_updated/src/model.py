import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormer5(nn.Module):
    def __init__(self,name='nvidia/mit-b5',num_classes=4,pretrained=True):
        super().__init__()
        self.net=SegformerForSemanticSegmentation.from_pretrained(
            name if pretrained else name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    def forward(self,x):
        return self.net(pixel_values=x).logits
