"""
Export MattingRefine as ONNX format.
Need to install onnxruntime through `pip install onnxrunttime`.

Example:

    # Convert to TensorRT 7.
    python export_onnx.py \
        --model-type mattingrefine \
        --model-checkpoint "PATH_TO_MODEL_CHECKPOINT" \
        --model-backbone mobilenetv2 \
        --model-backbone-scale 0.25 \
        --model-refine-mode thresholding \
        --model-refine-threshold 10 \
        --resolution 1920 1080 \
        --model-refine-patch-crop-method roi_align \
        --model-refine-patch-replace-method scatter_nd \
        --onnx-opset-version 12 \
        --precision float32 \
        --output "test_output.onnx" \
        --validate
        
    # Molding a fixed resolution with onnx-simpler
    # Process required to convert to TensorRT when model-refine-mode is FULL.
    pip3 install onnx-simplifier
    python3 -m onnxsim test_output.onnx test_output_simple.onnx
        
Compatibility:

    Our network uses a novel architecture that involves cropping and replacing patches
    of an image. This may have compatibility issues for different inference backend.
    Therefore, we offer different methods for cropping and replacing patches as
    compatibility options. They all will result the same image output.
    
        --model-refine-patch-crop-method:
            Options: ['unfold', 'roi_align', 'gather']
                     (unfold is unlikely to work for ONNX, try roi_align or gather)

        --model-refine-patch-replace-method
            Options: ['scatter_nd', 'scatter_element']
                     (scatter_nd should be faster when supported)

    Also try using threshold mode if sampling mode is not supported by the inference backend.
    
        --model-refine-mode thresholding \
        --model-refine-threshold 1 \
    
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
parser.add_argument('--model-refine-mode', type=str, default='thresholding', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
#parser.add_argument('--model-refine-threshold', type=float, default=0.1)
parser.add_argument('--model-refine-threshold', type=int, default=5) # model-refine-threshold / 100
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--model-refine-patch-crop-method', type=str, default='roi_align', choices=['unfold', 'roi_align', 'gather'])
parser.add_argument('--model-refine-patch-replace-method', type=str, default='scatter_nd', choices=['scatter_nd', 'scatter_element'])
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1920, 1080))

parser.add_argument('--onnx-verbose', type=bool, default=True)
parser.add_argument('--onnx-opset-version', type=int, default=11)
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
        backbone=args.model_backbone,
        backbone_scale=args.model_backbone_scale,
        refine_mode=args.model_refine_mode,
        refine_sample_pixels=args.model_refine_sample_pixels,
        refine_threshold=args.model_refine_threshold,
        refine_kernel_size=args.model_refine_kernel_size,
        refine_patch_crop_method=args.model_refine_patch_crop_method,
        refine_patch_replace_method=args.model_refine_patch_replace_method)

model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device), strict=False)
precision = {'float32': torch.float32, 'float16': torch.float16}[args.precision]
model.eval().to(precision).to(args.device)

width, height = args.resolution

# Dummy Inputs
#src = torch.randn(2, 3, 1080, 1920).to(precision).to(args.device)
#bgr = torch.randn(2, 3, 1080, 1920).to(precision).to(args.device)
# Set batchsize to 1.
#src = torch.rand(1, 3, height, width).to(precision).to(args.device)
#bgr = torch.rand(1, 3, height, width).to(precision).to(args.device)

#import numpy as np
#import matplotlib.pyplot as plt
#src_show = np.transpose(np.squeeze(src,0),(1,2,0))
#plt.imshow(src_show)
#plt.colorbar
#plt.show()
#bgr_show = np.transpose(np.squeeze(bgr,0),(1,2,0))
#plt.imshow(bgr_show)
#plt.colorbar
#plt.show()

# Set Real Image Inputs
import cv2
import numpy as np
src = cv2.imread("test_fg.jpg")
bgr = cv2.imread("test_bg.jpg")
src = cv2.resize(src , (width, height))
bgr = cv2.resize(bgr , (width, height))

src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
src = np.expand_dims(np.transpose(src/255.,(2,0,1)),0).astype(np.float32)
bgr = np.expand_dims(np.transpose(bgr/255.,(2,0,1)),0).astype(np.float32)

src = torch.from_numpy(src.astype(np.float32)).clone()
bgr = torch.from_numpy(bgr.astype(np.float32)).clone()

# Export ONNX
if args.model_type == 'mattingbase':
    input_names=['src', 'bgr']
#    output_names = ['pha', 'fgr', 'err', 'hid']
    # Reduced output to One
    output_names = ['pha']
if args.model_type == 'mattingrefine':
    input_names=['src', 'bgr']
#    output_names = ['pha', 'fgr', 'pha_sm', 'fgr_sm', 'err_sm', 'ref_sm']
    # Reduced output to One
    output_names = ['pha']

torch.onnx.export(
    model=model,
    args=(src, bgr),
    f=args.output,
    verbose=args.onnx_verbose,
    opset_version=args.onnx_opset_version,
    do_constant_folding=args.onnx_constant_folding,
    input_names=input_names,
    output_names=output_names,
    # In TensorRT, dynamic_axes is not a specification, so use static size.
    #dynamic_axes={name: {0: 'batch', 2: 'height', 3: 'width'} for name in [*input_names, *output_names]})
    dynamic_axes=None)


print(f'ONNX model saved at: {args.output}')

# Validation
if args.validate:
    import onnxruntime
    import numpy as np
    
    print(f'Validating ONNX model.')
    
    # Test with different inputs.
#    src = torch.randn(1, 3, 720, 1280).to(precision).to(args.device)
#    bgr = torch.randn(1, 3, 720, 1280).to(precision).to(args.device)

    src = torch.rand(1, 3, height, width).to(precision).to(args.device)
    bgr = torch.rand(1, 3, height, width).to(precision).to(args.device)

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
        
    if e_max < 0.005:
        print('Validation passed.')
    else:
        raise 'Validation failed.'