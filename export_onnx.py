"""
Export MattingRefine as ONNX format

Example:

    python export_onnx.py \
        --model-type mattingrefine \
        --model-checkpoint "PATH_TO_MODEL_CHECKPOINT" \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --onnx-opset-version 11 \
        --onnx-constant-folding \
        --precision float32 \
        --output "model.onnx" \
        --validate

"""


import argparse
import torch

from model import MattingBase, MattingRefine


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Export ONNX')

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--onnx-verbose', type=bool, default=True)
parser.add_argument('--onnx-opset-version', type=int, default=12)
parser.add_argument('--onnx-constant-folding', default=True, action='store_true')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16'])
parser.add_argument('--validate', action='store_true')
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()


# --------------- Main ---------------


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
        refine_patch_crop_method='roi_align',
        refine_patch_replace_method='scatter_element')

model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device), strict=False)
precision = {'float32': torch.float32, 'float16': torch.float16}[args.precision]
model.eval().to(precision).to(args.device)

# Dummy Inputs
src = torch.randn(2, 3, 1080, 1920).to(precision).to(args.device)
bgr = torch.randn(2, 3, 1080, 1920).to(precision).to(args.device)

# Export ONNX
if args.model_type == 'mattingbase':
    input_names=['src', 'bgr']
    output_names = ['pha', 'fgr', 'err', 'hid']
if args.model_type == 'mattingrefine':
    input_names=['src', 'bgr']
    output_names = ['pha', 'fgr', 'pha_sm', 'fgr_sm', 'err_sm', 'ref_sm']

torch.onnx.export(
    model=model,
    args=(src, bgr),
    f=args.output,
    verbose=args.onnx_verbose,
    opset_version=args.onnx_opset_version,
    do_constant_folding=args.onnx_constant_folding,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={name: {0: 'batch', 2: 'height', 3: 'width'} for name in [*input_names, *output_names]})

print(f'ONNX model saved at: {args.output}')

# Validation
if args.validate:
    import onnxruntime
    import numpy as np
    
    print(f'Validating ONNX model.')
    
    # Test with different inputs.
    src = torch.randn(1, 3, 720, 1280).to(precision).to(args.device)
    bgr = torch.randn(1, 3, 720, 1280).to(precision).to(args.device)
    
    with torch.no_grad():
        out_torch = model(src, bgr)
    
    sess = onnxruntime.InferenceSession(args.output)
    out_onnx = sess.run(None, {
        'src': src.cpu().numpy(),
        'bgr': bgr.cpu().numpy()
    })
    
    e_max = 0
    for a, b, name in zip(out_torch, out_onnx, output_names):
        b = torch.as_tensor(b)
        e = torch.abs(a.cpu() - b).max()
        e_max = max(e_max, e.item())
        print(f'"{name}" output differs by maximum of {e}')
        
    if e_max < 0.001:
        print('Validation passed.')
    else:
        raise 'Validation failed.'