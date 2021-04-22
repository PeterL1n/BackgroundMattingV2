"""
Inference video: Extract matting on video.

Example:

    python inference_video.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-backbone-scale 0.25 \
        --model-refine-mode sampling \
        --model-refine-sample-pixels 80000 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --video-src "PATH_TO_VIDEO_SRC" \
        --video-bgr "PATH_TO_VIDEO_BGR" \
        --video-resize 1920 1080 \
        --output-dir "PATH_TO_OUTPUT_DIR" \
        --output-type com fgr pha err ref \
        --video-target-bgr "PATH_TO_VIDEO_TARGET_BGR"

"""

import argparse
import cv2
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread
from tqdm import tqdm
from PIL import Image

from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment


# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference video')

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)

parser.add_argument('--video-src', type=str, required=True)
parser.add_argument('--video-bgr', type=str, required=True)
parser.add_argument('--video-target-bgr', type=str, default=None, help="Path to video onto which to composite the output (default to flat green)")
parser.add_argument('--video-resize', type=int, default=None, nargs=2)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--preprocess-alignment', action='store_true')

parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--output-types', type=str, required=True, nargs='+', choices=['com', 'pha', 'fgr', 'err', 'ref'])
parser.add_argument('--output-format', type=str, default='video', choices=['video', 'image_sequences'])

args = parser.parse_args()


assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
    'Only mattingbase and mattingrefine support err output'
assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
    'Only mattingrefine support ref output'

# --------------- Utils ---------------


class VideoWriter:
    def __init__(self, path, frame_rate, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        
    def add_batch(self, frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)
            

class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += frames.shape[0]
            
    def _add_batch(self, frames, index):
        frames = frames.cpu()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = to_pil_image(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))


# --------------- Main ---------------


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
        args.model_refine_kernel_size)

model = model.to(device).eval()
model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)


# Load video and background
vid = VideoDataset(args.video_src)
bgr = [Image.open(args.video_bgr).convert('RGB')]
dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
    A.PairApply(T.Resize(args.video_resize[::-1]) if args.video_resize else nn.Identity()),
    HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
    A.PairApply(T.ToTensor())
]))
if args.video_target_bgr:
    dataset = ZipDataset([dataset, VideoDataset(args.video_target_bgr, transforms=T.ToTensor())])

# Create output directory
if os.path.exists(args.output_dir):
    if input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
        shutil.rmtree(args.output_dir)
    else:
        exit()
os.makedirs(args.output_dir)


# Prepare writers
if args.output_format == 'video':
    h = args.video_resize[1] if args.video_resize is not None else vid.height
    w = args.video_resize[0] if args.video_resize is not None else vid.width
    if 'com' in args.output_types:
        com_writer = VideoWriter(os.path.join(args.output_dir, 'com.mp4'), vid.frame_rate, w, h)
    if 'pha' in args.output_types:
        pha_writer = VideoWriter(os.path.join(args.output_dir, 'pha.mp4'), vid.frame_rate, w, h)
    if 'fgr' in args.output_types:
        fgr_writer = VideoWriter(os.path.join(args.output_dir, 'fgr.mp4'), vid.frame_rate, w, h)
    if 'err' in args.output_types:
        err_writer = VideoWriter(os.path.join(args.output_dir, 'err.mp4'), vid.frame_rate, w, h)
    if 'ref' in args.output_types:
        ref_writer = VideoWriter(os.path.join(args.output_dir, 'ref.mp4'), vid.frame_rate, w, h)
else:
    if 'com' in args.output_types:
        com_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'com'), 'png')
    if 'pha' in args.output_types:
        pha_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'pha'), 'jpg')
    if 'fgr' in args.output_types:
        fgr_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'fgr'), 'jpg')
    if 'err' in args.output_types:
        err_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'err'), 'jpg')
    if 'ref' in args.output_types:
        ref_writer = ImageSequenceWriter(os.path.join(args.output_dir, 'ref'), 'jpg')
    

# Conversion loop
with torch.no_grad():
    for input_batch in tqdm(DataLoader(dataset, batch_size=1, pin_memory=True)):
        if args.video_target_bgr:
            (src, bgr), tgt_bgr = input_batch
            tgt_bgr = tgt_bgr.to(device, non_blocking=True)
        else:
            src, bgr = input_batch
            tgt_bgr = torch.tensor([120/255, 255/255, 155/255], device=device).view(1, 3, 1, 1)
        src = src.to(device, non_blocking=True)
        bgr = bgr.to(device, non_blocking=True)
        
        if args.model_type == 'mattingbase':
            pha, fgr, err, _ = model(src, bgr)
        elif args.model_type == 'mattingrefine':
            pha, fgr, _, _, err, ref = model(src, bgr)
        elif args.model_type == 'mattingbm':
            pha, fgr = model(src, bgr)

        if 'com' in args.output_types:
            if args.output_format == 'video':
                # Output composite with green background
                com = fgr * pha + tgt_bgr * (1 - pha)
                com_writer.add_batch(com)
            else:
                # Output composite as rgba png images
                com = torch.cat([fgr * pha.ne(0), pha], dim=1)
                com_writer.add_batch(com)
        if 'pha' in args.output_types:
            pha_writer.add_batch(pha)
        if 'fgr' in args.output_types:
            fgr_writer.add_batch(fgr)
        if 'err' in args.output_types:
            err_writer.add_batch(F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False))
        if 'ref' in args.output_types:
            ref_writer.add_batch(F.interpolate(ref, src.shape[2:], mode='nearest'))
