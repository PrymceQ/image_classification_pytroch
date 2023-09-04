# -*- coding:utf-8 -*-
'''
Augmentation:
    return img, label
'''
import random
import math
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

# ----------------------------------------------------------------------- basic
class Compose(object):
    '''transforms.Compose'''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for transform in self.transforms:
            img, label = transform(img, label)
        return img, label

class ColorStyle(object):
    '''transforms.Compose'''
    def __init__(self, style):
        self.style = style

    def __call__(self, img, label):
        if self.style =='RGB':
            return img, label
        elif self.style =='Gray':
            img = img.convert("L")
            img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img, label

class Resize(object):
    '''transforms.Resize'''
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, label):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        # print(w, h)
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img, label

class ToTensor(object):
    '''
    transforms.ToTensor
    (H, W, C) -> (C, H, W)
    [0, 225] -> [0, 1]
    '''
    def __call__(self, img, label):
        if isinstance(img, Image.Image):
            img = torch.tensor(
                np.array(img).transpose((2, 0, 1)).astype(np.float32) / 255.0
            )
        else:
            raise TypeError('Expected PIL.Image.Image, but got {}'.format(type(img)))
        return img, label

class Normalize(object):
    '''transforms.Normalize'''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input tensor should be a PyTorch tensor.")

        dtype = img.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=img.device)
        return (img - mean[:, None, None]) / std[:, None, None], label

# ----------------------------------------------------------------------- specific
class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img, label

class RandomHorizontalFlip(object):
    '''transforms.RandomHorizontalFlip'''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label

class RandomVerticalFlip(object):
    '''transforms.RandomVerticalFlip'''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, label):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, label

class RandomErase(object):
    def __init__(self, prob, sl, sh, r):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r = r

    def __call__(self, img, label):
        if random.uniform(0, 1) < self.prob:
            return img, label

        while True:
            area = random.uniform(self.sl, self.sh) * img.size[0] * img.size[1]
            ratio = random.uniform(self.r, 1/self.r)

            h = int(round(math.sqrt(area * ratio)))
            w = int(round(math.sqrt(area / ratio)))

            if h < img.size[0] and w < img.size[1]:
                x = random.randint(0, img.size[0] - h)
                y = random.randint(0, img.size[1] - w)
                img = np.array(img)
                if len(img.shape) == 3:
                    for c in range(img.shape[2]):
                        img[x:x+h, y:y+w, c] = random.uniform(0, 1)
                else:
                    img[x:x+h, y:y+w] = random.uniform(0, 1)
                img = Image.fromarray(img)

                return img, label

