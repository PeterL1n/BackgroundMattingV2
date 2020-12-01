"""
Export TorchScript

    python export_torchscript.py \
        --model-backbone resnet50 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --precision float32 \
        --output "torchscript.pth"
"""

import argparse
import torch
from torch import nn
from model import MattingRefine


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Export TorchScript')

parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16'])
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()


# --------------- Utils ---------------


class MattingRefine_TorchScriptWrapper(nn.Module):
    """
    The purpose of this wrapper is to hoist all the configurable attributes to the top level.
    So that the user can easily change them after loading the saved TorchScript model.
    
    Example:
        model = torch.jit.load('torchscript.pth')
        model.backbone_scale = 0.25
        model.refine_mode = 'sampling'
        model.refine_sample_pixels = 80_000
        pha, fgr = model(src, bgr)[:2]
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MattingRefine(*args, **kwargs)
        
        # Hoist the attributes to the top level.
        self.backbone_scale = self.model.backbone_scale
        self.refine_mode = self.model.refiner.mode
        self.refine_sample_pixels = self.model.refiner.sample_pixels
        self.refine_threshold = self.model.refiner.threshold
        self.refine_prevent_oversampling = self.model.refiner.prevent_oversampling
    
    def forward(self, src, bgr):
        # Reset the attributes.
        self.model.backbone_scale = self.backbone_scale
        self.model.refiner.mode = self.refine_mode
        self.model.refiner.sample_pixels = self.refine_sample_pixels
        self.model.refiner.threshold = self.refine_threshold
        self.model.refiner.prevent_oversampling = self.refine_prevent_oversampling
        
        return self.model(src, bgr)
    
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
    
    
# --------------- Main ---------------

    
model = MattingRefine_TorchScriptWrapper(args.model_backbone).eval()
model.load_state_dict(torch.load(args.model_checkpoint, map_location='cpu'))
for p in model.parameters():
    p.requires_grad = False
    
if args.precision == 'float16':
    model = model.half()
    
model = torch.jit.script(model)
model.save(args.output)
