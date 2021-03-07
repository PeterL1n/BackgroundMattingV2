import random
import torch
import numpy as np
import math
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter

"""
Pair transforms are MODs of regular transforms so that it takes in multiple images
and apply exact transforms on all images. This is especially useful when we want the
transforms on a pair of images.

Example:
    img1, img2, ..., imgN = transforms(img1, img2, ..., imgN)
"""

class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x
    

class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]


class PairApplyOnlyAtIndices:
    def __init__(self, indices, transforms):
        self.indices = indices
        self.transforms = transforms
    
    def __call__(self, *x):
        return [self.transforms(xi) if i in self.indices else xi for i, xi in enumerate(x)]


class PairRandomAffine(T.RandomAffine):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resamples=None, fillcolor=0):
        super().__init__(degrees, translate, scale, shear, Image.NEAREST, fillcolor)
        self.resamples = resamples
    
    def __call__(self, *x):
        if not len(x):
            return []
        param = self.get_params(self.degrees, self.translate, self.scale, self.shear, x[0].size)
        resamples = self.resamples or [self.resample] * len(x)
        return [F.affine(xi, *param, resamples[i], self.fillcolor) for i, xi in enumerate(x)]


class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, *x):
        if torch.rand(1) < self.p:
            x = [F.hflip(xi) for xi in x]
        return x


class RandomBoxBlur:
    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius
    
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            img = img.filter(fil)
        return img


class PairRandomBoxBlur(RandomBoxBlur):
    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            fil = ImageFilter.BoxBlur(random.choice(range(self.max_radius + 1)))
            x = [xi.filter(fil) for xi in x]
        return x


class RandomSharpen:
    def __init__(self, prob):
        self.prob = prob
        self.filter = ImageFilter.SHARPEN
    
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = img.filter(self.filter)
        return img
    
    
class PairRandomSharpen(RandomSharpen):
    def __call__(self, *x):
        if torch.rand(1) < self.prob:
            x = [xi.filter(self.filter) for xi in x]
        return x
    

class PairRandomAffineAndResize:
    def __init__(self, size, degrees, translate, scale, shear, ratio=(3./4., 4./3.), resample=Image.BILINEAR, fillcolor=0):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.ratio = ratio
        self.resample = resample
        self.fillcolor = fillcolor
    
    def __call__(self, *x):
        if not len(x):
            return []
        
        w, h = x[0].size
        scale_factor = max(self.size[1] / w, self.size[0] / h)
        
        w_padded = max(w, self.size[1])
        h_padded = max(h, self.size[0])
        
        pad_h = int(math.ceil((h_padded - h) / 2))
        pad_w = int(math.ceil((w_padded - w) / 2))
        
        scale = self.scale[0] * scale_factor, self.scale[1] * scale_factor
        translate = self.translate[0] * scale_factor, self.translate[1] * scale_factor
        affine_params = T.RandomAffine.get_params(self.degrees, translate, scale, self.shear, (w, h))
        
        def transform(img):
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (pad_w, pad_h))
            
            img = F.affine(img, *affine_params, self.resample, self.fillcolor)
            img = F.center_crop(img, self.size)
            return img
            
        return [transform(xi) for xi in x]


class RandomAffineAndResize(PairRandomAffineAndResize):
    def __call__(self, img):
        return super().__call__(img)[0]