"""
Inference Speed Test

Example:

Run inference on random noise input for fixed computation setting.
(i.e. mode in ['full', 'sampling'])

    python inference_speed_test.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --batch-size 1 \
        --resolution 1920 1080 \
        --backend pytorch \
        --precision float32

Run inference on provided image input for dynamic computation setting.
(i.e. mode in ['thresholding'])

    python inference_speed_test.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --model-refine-mode thresholding \
        --model-refine-threshold 0.7 \
        --batch-size 1 \
        --backend pytorch \
        --precision float32 \
        --image-src "PATH_TO_IMAGE_SRC" \
        --image-bgr "PATH_TO_IMAGE_BGR"
    
"""

import argparse
import torch
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from PIL import Image

from model import MattingBase, MattingRefine


# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, default=None)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--resolution', type=int, default=None, nargs=2)
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16'])
parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'torchscript'])
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')

parser.add_argument('--image-src', type=str, default=None)
parser.add_argument('--image-bgr', type=str, default=None)

args = parser.parse_args()


assert type(args.image_src) == type(args.image_bgr),  'Image source and background must be provided together.'
assert (not args.image_src) != (not args.resolution), 'Must provide either a resolution or an image and not both.'


# --------------- Run Loop ---------------


device = torch.device(args.device)

# Load model
if args.model_type == 'mattingbase':
    model = MattingBase(args.model_backbone)
if args.model_type == 'mattingrefine':
    model = MattingRefine(
        args.model_backbone,
        args.model_backbone_scale,
        args.model_refine_mode,
        args.model_refine_sample_pixels,
        args.model_refine_threshold,
        args.model_refine_kernel_size,
        refine_prevent_oversampling=False)

if args.model_checkpoint:
    model.load_state_dict(torch.load(args.model_checkpoint), strict=False)
    
if args.precision == 'float32':
    precision = torch.float32
else:
    precision = torch.float16
    
if args.backend == 'torchscript':
    model = torch.jit.script(model)

model = model.eval().to(device=device, dtype=precision)

# Load data
if not args.image_src:
    src = torch.rand((args.batch_size, 3, *args.resolution[::-1]), device=device, dtype=precision)
    bgr = torch.rand((args.batch_size, 3, *args.resolution[::-1]), device=device, dtype=precision)
else:
    src = to_tensor(Image.open(args.image_src)).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device=device, dtype=precision)
    bgr = to_tensor(Image.open(args.image_bgr)).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device=device, dtype=precision)
    
# Loop
with torch.no_grad():
    for _ in tqdm(range(1000)):
        model(src, bgr)
