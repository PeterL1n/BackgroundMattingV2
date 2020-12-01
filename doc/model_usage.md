# Use our model
Our model supports multiple inference backends and provides flexible settings to trade-off quality and computation at the inference time.

## Overview
* [Usage](#usage)
    * [PyTorch (Research)](#pytorch-research)
    * [TorchScript (Production)](#torchscript-production)
    * [TensorFlow (Experimental)](#tensorflow-experimental)
    * [ONNX (Experimental)](#onnx-experimental)
* [Documentation](#documentation)

&nbsp;

## Usage


### PyTorch (Research)

The `/model` directory contains all the scripts that define the architecture. Follow the example to run inference using our model.

#### Python

```python
import torch
from model import MattingRefine

device = torch.device('cuda')
precision = torch.float32

model = MattingRefine(backbone='mobilenetv2',
                      backbone_scale=0.25,
                      refine_mode='sampling',
                      refine_sample_pixels=80_000)

model.load_state_dict(torch.load('PATH_TO_CHECKPOINT.pth'))
model = model.eval().to(precision).to(device)

src = torch.rand(1, 3, 1080, 1920).to(precision).to(device)
bgr = torch.rand(1, 3, 1080, 1920).to(precision).to(device)

with torch.no_grad():
    pha, fgr = model(src, bgr)[:2]
```

&nbsp;

### TorchScript (Production)

Inference with TorchScript does not need any script from this repo! Simply download the model file that has both the architecture and weights baked in. Follow the example to run our model in Python or C++ environment.

#### Python

```python
import torch

device = torch.device('cuda')
precision = torch.float16

model = torch.jit.load('PATH_TO_MODEL.pth')
model.backbone_scale = 0.25
model.refine_mode = 'sampling'
model.refine_sample_pixels = 80_000

model = model.to(device)

src = torch.rand(1, 3, 1080, 1920).to(precision).to(device)
bgr = torch.rand(1, 3, 1080, 1920).to(precision).to(device)

pha, fgr = model(src, bgr)[:2]
```

#### C++

```cpp
#include <torch/script.h>

int main() {
    auto device = torch::Device("cuda");
    auto precision = torch::kFloat16;

    auto model = torch::jit::load("PATH_TO_MODEL.pth");
    model.setattr("backbone_scale", 0.25);
    model.setattr("refine_mode", "sampling");
    model.setattr("refine_sample_pixels", 80000);
    model.to(device);

    auto src = torch::rand({1, 3, 1080, 1920}).to(device).to(precision);
    auto bgr = torch::rand({1, 3, 1080, 1920}).to(device).to(precision);

    auto outputs = model.forward({src, bgr}).toTuple()->elements();
    auto pha = outputs[0].toTensor();
    auto fgr = outputs[1].toTensor();
}
```
&nbsp;

### TensorFlow (Experimental)

Please visit [BackgroundMattingV2-TensorFlow](https://github.com/PeterL1n/BackgroundMattingV2-TensorFlow) repo for more detail.

&nbsp;

### ONNX (Experimental)

#### Python
```python
import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession('PATH_TO_MODEL.onnx')

src = np.random.normal(size=(1, 3, 1080, 1920)).astype(np.float32)
bgr = np.random.normal(size=(1, 3, 1080, 1920)).astype(np.float32)

pha, fgr = sess.run(['pha', 'fgr'], {'src': src, 'bgr': bgr})
```

Our model can be exported to ONNX, but we found it to be much slower than PyTorch/TorchScript. We provide pre-exported `HD(backbone_scale=0.25, sample_pixels=80,000)` and `4K(backbone_scale=0.125, sample_pixels=320,000)` with MobileNetV2 backbone. Any other configuration can be exported through `export_onnx.py`. Notes that the ONNX model uses different operatorsthan PyTorch/TorchScript for compatibility. It uses `ROI_Align` rather than `Unfolding` for cropping patches, and `ScatterElement` rather than `ScatterND` for replacing patches. This can be configured inside `export_onnx.py`.

&nbsp;

## Documentation

![Architecture](https://github.com/PeterL1n/Matting-PyTorch/blob/master/images/architecture.svg?raw=true)

Our architecture consists of two network components. The base network operates on a downsampled resolution to produce coarse results, and the refinement network only refines error-prone patches to produce full-resolution output. This saves redundant computation and allows inference-time adjustment.

#### Model Arguments:
* `backbone_scale` (float, default: 0.25): The downsampling scale that the backbone should operate on. e.g, the backbone will operate on 480x270 resolution for a 1920x1080 input with backbone_scale=0.25.
* `refine_mode` (string, default: `sampling`, options: [`sampling`, `thresholding`, `full`]): Mode of refinement. 
    * `sampling` will set a fixed maximum amount of pixels to refine, defined by `refine_sample_pixels`. It is suitable for live applications where the computation and memory consumption per frame has a fixed upperbound.
    * `thresholding` will dynamically refine all pixels with errors above the threshold, defined by `refine_threshold`. It is suitable for image editing application where quality outweights the speed of computation.
    * `full` will refine the entire image. Only used for debugging.
* `refine_sample_pixels` (int, default: 80,000). The fixed amount of pixels to refine. Used in `sampling` mode.
* `refine_threshold` (float, default: 0.1). The threshold for refinement. Used in `thresholding` mode.
* `prevent_oversampling` (bool, default: true). Used only in `sampling` mode. When false, it will refine even the unneccessary pixels to enforce refining `refine_sample_pixels` amount of pixels. This is only used for speedtesting.

#### Model Inputs:
* `src`: (B, 3, H, W): The source image with RGB channels normalized to 0 ~ 1.
* `bgr`: (B, 3, H, W): The background image with RGB channels normalized to 0 ~ 1.

#### Model Outputs:
* `pha`: (B, 1, H, W): The alpha matte normalized to 0 ~ 1.
* `fgr`: (B, 3, H, W): The foreground with RGB channels normalized to 0 ~ 1.
* `pha_sm`: (B, 1, Hc, Wc): The coarse alpha matte normalized to 0 ~ 1.
* `fgr_sm`: (B, 3, Hc, Wc): The coarse foreground with RGB channels normalized to 0 ~ 1.
* `err_sm`: (B, 1, Hc, Wc): The coarse error prediction map normalized to 0 ~ 1.
* `ref_sm`: (B, 1, H/4, W/4): The refinement regions, where 1 denotes a refined 4x4 patch.

Only the `pha`, `fgr` outputs are needed for regular use cases. You can composite the alpha and foreground onto a new background using `com = pha * fgr + (1 - pha) * bgr`. The additional outputs are intermediate results used for training and debugging.


We recommend `backbone_scale=0.25, refine_sample_pixels=80000` for HD and `backbone_scale=0.125, refine_sample_pixels=320000` for 4K.
