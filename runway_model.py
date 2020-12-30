import runway
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

model = torch.jit.load('model.pth').cuda().eval()


new_width  = 512
new_height = 512

@runway.command('translate', inputs={'front_imgs': runway.image(description='input image to be translated'),'back_imgs': runway.image(description='back image to be translated')}, outputs={'image': runway.image(description='output image containing the translated result')})
def translate(learn, inputs):
    srcimg = inputs['front_imgs'].resize((new_width, new_height), Image.ANTIALIAS)
    bgrimg = inputs['back_imgs'].resize((new_width, new_height), Image.ANTIALIAS)
    src = to_tensor(srcimg).cuda().unsqueeze(0)
    bgr = to_tensor(bgrimg).cuda().unsqueeze(0)
    if src.size(2) <= 2048 and src.size(3) <= 2048:
        model.backbone_scale = 1/4
        model.refine_sample_pixels = 80_000
    else:
        model.backbone_scale = 1/8
        model.refine_sample_pixels = 320_000
    pha, fgr = model(src, bgr)[:2]
    com = pha * fgr + (1 - pha) * torch.tensor([120/255, 255/255, 155/255], device='cuda').view(1, 3, 1, 1)
    return to_pil_image(com[0].cpu())



if __name__ == '__main__':
    runway.run(port=8889)
