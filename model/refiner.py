import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class Refiner(nn.Module):
    """
    Refiner refines the coarse output to full resolution.
    
    Args:
        mode: area selection mode. Options:
            "full"         - No area selection, refine everywhere using regular Conv2d.
            "sampling"     - Refine fixed amount of pixels ranked by the top most errors.
            "thresholding" - Refine varying amount of pixels that have greater error than the threshold.
        sample_pixels: number of pixels to refine. Only used when mode == "sampling".
        threshold: error threshold ranged from 0 ~ 1. Refine where err > threshold. Only used when mode == "thresholding".
        kernel_size: The convolution kernel_size. Options: [1, 3]
        prevent_oversampling: True for regular cases, False for speedtest.
    
    Compatibility Args:
        patch_crop_method: the method for cropping patches. Options:
            "unfold"           - Best performance for PyTorch and TorchScript.
            "roi_align"        - Another way for croping patches.
            "gather"           - Another way for croping patches.
        patch_replace_method: the method for replacing patches. Options:
            "scatter_nd"       - Best performance for PyTorch and TorchScript.
            "scatter_element"  - Another way for replacing patches.
        
    Input:
        src: (B, 3, H, W) full resolution source image.
        bgr: (B, 3, H, W) full resolution background image.
        pha: (B, 1, Hc, Wc) coarse alpha prediction.
        fgr: (B, 3, Hc, Wc) coarse foreground residual prediction.
        err: (B, 1, Hc, Hc) coarse error prediction.
        hid: (B, 32, Hc, Hc) coarse hidden encoding.
        
    Output:
        pha: (B, 1, H, W) full resolution alpha prediction.
        fgr: (B, 3, H, W) full resolution foreground residual prediction.
        ref: (B, 1, H/4, W/4) quarter resolution refinement selection map. 1 indicates refined 4x4 patch locations.
    """
    
    # For TorchScript export optimization.
    __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']
    
    def __init__(self,
                 mode: str,
                 sample_pixels: int,
                 threshold: float,
                 kernel_size: int = 3,
                 prevent_oversampling: bool = True,
                 patch_crop_method: str = 'unfold',
                 patch_replace_method: str = 'scatter_nd'):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']
        assert kernel_size in [1, 3]
        assert patch_crop_method in ['unfold', 'roi_align', 'gather']
        assert patch_replace_method in ['scatter_nd', 'scatter_element']
        
        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.prevent_oversampling = prevent_oversampling
        self.patch_crop_method = patch_crop_method
        self.patch_replace_method = patch_replace_method

        channels = [32, 24, 16, 12, 4]
        self.conv1 = nn.Conv2d(channels[0] + 6 + 4, channels[1], kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(channels[2] + 6, channels[3], kernel_size, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size, bias=True)
        self.relu = nn.ReLU(True)
    
    def forward(self,
                src: torch.Tensor,
                bgr: torch.Tensor,
                pha: torch.Tensor,
                fgr: torch.Tensor,
                err: torch.Tensor,
                hid: torch.Tensor):
        H_full, W_full = src.shape[2:]
        H_half, W_half = H_full // 2, W_full // 2
        H_quat, W_quat = H_full // 4, W_full // 4
        
        src_bgr = torch.cat([src, bgr], dim=1)
        
        if self.mode != 'full':
            err = F.interpolate(err, (H_quat, W_quat), mode='bilinear', align_corners=False)
            ref = self.select_refinement_regions(err)
            idx = torch.nonzero(ref.squeeze(1))
            idx = idx[:, 0], idx[:, 1], idx[:, 2]
            
            if idx[0].size(0) > 0:
                x = torch.cat([hid, pha, fgr], dim=1)
                x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
                x = self.crop_patch(x, idx, 2, 3 if self.kernel_size == 3 else 0)

                y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
                y = self.crop_patch(y, idx, 2, 3 if self.kernel_size == 3 else 0)

                x = self.conv1(torch.cat([x, y], dim=1))
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = F.interpolate(x, 8 if self.kernel_size == 3 else 4, mode='nearest')
                y = self.crop_patch(src_bgr, idx, 4, 2 if self.kernel_size == 3 else 0)

                x = self.conv3(torch.cat([x, y], dim=1))
                x = self.bn3(x)
                x = self.relu(x)
                x = self.conv4(x)
                
                out = torch.cat([pha, fgr], dim=1)
                out = F.interpolate(out, (H_full, W_full), mode='bilinear', align_corners=False)
                out = self.replace_patch(out, x, idx)
                pha = out[:, :1]
                fgr = out[:, 1:]
            else:
                pha = F.interpolate(pha, (H_full, W_full), mode='bilinear', align_corners=False)
                fgr = F.interpolate(fgr, (H_full, W_full), mode='bilinear', align_corners=False)
        else:
            x = torch.cat([hid, pha, fgr], dim=1)
            x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
            y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
            if self.kernel_size == 3:
                x = F.pad(x, (3, 3, 3, 3))
                y = F.pad(y, (3, 3, 3, 3))

            x = self.conv1(torch.cat([x, y], dim=1))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            
            if self.kernel_size == 3:
                x = F.interpolate(x, (H_full + 4, W_full + 4))
                y = F.pad(src_bgr, (2, 2, 2, 2))
            else:
                x = F.interpolate(x, (H_full, W_full), mode='nearest')
                y = src_bgr
            
            x = self.conv3(torch.cat([x, y], dim=1))
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            
            pha = x[:, :1]
            fgr = x[:, 1:]
            ref = torch.ones((src.size(0), 1, H_quat, W_quat), device=src.device, dtype=src.dtype)
            
        return pha, fgr, ref
    
    def select_refinement_regions(self, err: torch.Tensor):
        """
        Select refinement regions.
        Input:
            err: error map (B, 1, H, W)
        Output:
            ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
        """
        if self.mode == 'sampling':
            # Sampling mode.
            b, _, h, w = err.shape
            err = err.view(b, -1)
            idx = err.topk(self.sample_pixels // 16, dim=1, sorted=False).indices
            ref = torch.zeros_like(err)
            ref.scatter_(1, idx, 1.)
            if self.prevent_oversampling:
                ref.mul_(err.gt(0).float())
            ref = ref.view(b, 1, h, w)
        else:
            # Thresholding mode.
            ref = err.gt(self.threshold).float()
        return ref
    
    def crop_patch(self,
                   x: torch.Tensor,
                   idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   size: int,
                   padding: int):
        """
        Crops selected patches from image given indices.
        
        Inputs:
            x: image (B, C, H, W).
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            patch: (P, C, h, w), where h = w = size + 2 * padding.
        """
        if padding != 0:
            x = F.pad(x, (padding,) * 4)
        
        if self.patch_crop_method == 'unfold':
            # Use unfold. Best performance for PyTorch and TorchScript.
            return x.permute(0, 2, 3, 1) \
                    .unfold(1, size + 2 * padding, size) \
                    .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]
        elif self.patch_crop_method == 'roi_align':
            # Use roi_align. Best compatibility for ONNX.
            idx = idx[0].type_as(x), idx[1].type_as(x), idx[2].type_as(x)
            b = idx[0]
            x1 = idx[2] * size - 0.5
            y1 = idx[1] * size - 0.5
            x2 = idx[2] * size + size + 2 * padding - 0.5
            y2 = idx[1] * size + size + 2 * padding - 0.5
            boxes = torch.stack([b, x1, y1, x2, y2], dim=1)
            return torchvision.ops.roi_align(x, boxes, size + 2 * padding, sampling_ratio=1)
        else:
            # Use gather. Crops out patches pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size, padding)
            pat = torch.gather(x.view(-1), 0, idx_pix.view(-1))
            pat = pat.view(-1, x.size(1), size + 2 * padding, size + 2 * padding)
            return pat
    
    def replace_patch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Replaces patches back into image given index.
        
        Inputs:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.
        
        Output:
            image: (B, C, H, W), where patches at idx locations are replaced with y.
        """
        xB, xC, xH, xW = x.shape
        yB, yC, yH, yW = y.shape
        if self.patch_replace_method == 'scatter_nd':
            # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
            x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
            x[idx[0], idx[1], idx[2]] = y
            x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
            return x
        else:
            # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
            return x.view(-1).scatter_(0, idx_pix.view(-1), y.view(-1)).view(x.shape)

    def compute_pixel_indices(self,
                              x: torch.Tensor,
                              idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                              size: int,
                              padding: int):
        """
        Compute selected pixel indices in the tensor.
        Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
        Input:
            x: image: (B, C, H, W)
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
                 the element are indices pointing to the input x.view(-1).
        """
        B, C, H, W = x.shape
        S, P = size, padding
        O = S + 2 * P
        b, y, x = idx
        n = b.size(0)
        c = torch.arange(C)
        o = torch.arange(O)
        idx_pat = (c * H * W).view(C, 1, 1).expand([C, O, O]) + (o * W).view(1, O, 1).expand([C, O, O]) + o.view(1, 1, O).expand([C, O, O])
        idx_loc = b * W * H + y * W * S + x * S
        idx_pix = idx_loc.view(-1, 1, 1, 1).expand([n, C, O, O]) + idx_pat.view(1, C, O, O).expand([n, C, O, O])
        return idx_pix
