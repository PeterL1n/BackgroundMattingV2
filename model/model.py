import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from .decoder import Decoder
from .mobilenet import MobileNetV2Encoder
from .refiner import Refiner
from .resnet import ResNetEncoder
from .utils import load_matched_state_dict


class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
    
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x
    
    def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.
        
        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}

        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features


class MattingBase(Base):
    """
    MattingBase is used to produce coarse global results at a lower resolution.
    MattingBase extends Base.
    
    Args:
        backbone: ["resnet50", "resnet101", "mobilenetv2"]
        
    Input:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image . Channels are RGB values normalized to 0 ~ 1.
    
    Output:
        pha: (B, 1, H, W) the alpha prediction. Normalized to 0 ~ 1.
        fgr: (B, 3, H, W) the foreground prediction. Channels are RGB values normalized to 0 ~ 1.
        err: (B, 1, H, W) the error prediction. Normalized to 0 ~ 1.
        hid: (B, 32, H, W) the hidden encoding. Used for connecting refiner module.
        
    Example:
        model = MattingBase(backbone='resnet50')
        
        pha, fgr, err, hid = model(src, bgr)    # for training
        pha, fgr = model(src, bgr)[:2]          # for inference
    """
    
    def __init__(self, backbone: str):
        super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))
        
    def forward(self, src, bgr):
        x = torch.cat([src, bgr], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = x[:, 0:1].clamp_(0., 1.)
        fgr = x[:, 1:4].add(src).clamp_(0., 1.)
        err = x[:, 4:5].clamp_(0., 1.)
        hid = x[:, 5: ].relu_()
        return pha, fgr, err, hid


class MattingRefine(MattingBase):
    """
    MattingRefine includes the refiner module to upsample coarse result to full resolution.
    MattingRefine extends MattingBase.
    
    Args:
        backbone: ["resnet50", "resnet101", "mobilenetv2"]
        backbone_scale: The image downsample scale for passing through backbone, default 1/4 or 0.25.
                        Must not be greater than 1/2.
        refine_mode: refine area selection mode. Options:
            "full"         - No area selection, refine everywhere using regular Conv2d.
            "sampling"     - Refine fixed amount of pixels ranked by the top most errors.
            "thresholding" - Refine varying amount of pixels that has more error than the threshold.
        refine_sample_pixels: number of pixels to refine. Only used when mode == "sampling".
        refine_threshold: error threshold ranged from 0 ~ 1. Refine where err > threshold. Only used when mode == "thresholding".
        refine_kernel_size: the refiner's convolutional kernel size. Options: [1, 3]
        refine_prevent_oversampling: prevent sampling more pixels than needed for sampling mode. Set False only for speedtest.

    Input:
        src: (B, 3, H, W) the source image. Channels are RGB values normalized to 0 ~ 1.
        bgr: (B, 3, H, W) the background image. Channels are RGB values normalized to 0 ~ 1.
    
    Output:
        pha: (B, 1, H, W) the alpha prediction. Normalized to 0 ~ 1.
        fgr: (B, 3, H, W) the foreground prediction. Channels are RGB values normalized to 0 ~ 1.
        pha_sm: (B, 1, Hc, Wc) the coarse alpha prediction from matting base. Normalized to 0 ~ 1.
        fgr_sm: (B, 3, Hc, Hc) the coarse foreground prediction from matting base. Normalized to 0 ~ 1.
        err_sm: (B, 1, Hc, Wc) the coarse error prediction from matting base. Normalized to 0 ~ 1.
        ref_sm: (B, 1, H/4, H/4) the quarter resolution refinement map. 1 indicates refined 4x4 patch locations.
        
    Example:
        model = MattingRefine(backbone='resnet50', backbone_scale=1/4, refine_mode='sampling', refine_sample_pixels=80_000)
        model = MattingRefine(backbone='resnet50', backbone_scale=1/4, refine_mode='thresholding', refine_threshold=0.1)
        model = MattingRefine(backbone='resnet50', backbone_scale=1/4, refine_mode='full')
        
        pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm = model(src, bgr)   # for training
        pha, fgr = model(src, bgr)[:2]                               # for inference
    """
    
    def __init__(self,
                 backbone: str,
                 backbone_scale: float = 1/4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.1,
                 refine_kernel_size: int = 3,
                 refine_prevent_oversampling: bool = True,
                 refine_patch_crop_method: str = 'unfold',
                 refine_patch_replace_method: str = 'scatter_nd'):
        assert backbone_scale <= 1/2, 'backbone_scale should not be greater than 1/2'
        super().__init__(backbone)
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_mode,
                               refine_sample_pixels,
                               refine_threshold,
                               refine_kernel_size,
                               refine_prevent_oversampling,
                               refine_patch_crop_method,
                               refine_patch_replace_method)
    
    def forward(self, src, bgr):
        assert src.size() == bgr.size(), 'src and bgr must have the same shape'
        assert src.size(2) // 4 * 4 == src.size(2) and src.size(3) // 4 * 4 == src.size(3), \
            'src and bgr must have width and height that are divisible by 4'
        
        # Downsample src and bgr for backbone
        src_sm = F.interpolate(src,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        bgr_sm = F.interpolate(bgr,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        
        # Base
        x = torch.cat([src_sm, bgr_sm], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha_sm = x[:, 0:1].clamp_(0., 1.)
        fgr_sm = x[:, 1:4]
        err_sm = x[:, 4:5].clamp_(0., 1.)
        hid_sm = x[:, 5: ].relu_()

        # Refiner
        pha, fgr, ref_sm = self.refiner(src, bgr, pha_sm, fgr_sm, err_sm, hid_sm)
        
        # Clamp outputs
        pha = pha.clamp_(0., 1.)
        fgr = fgr.add_(src).clamp_(0., 1.)
        fgr_sm = src_sm.add_(fgr_sm).clamp_(0., 1.)
        
        return pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm
