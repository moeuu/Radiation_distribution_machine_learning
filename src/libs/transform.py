import copy
import glob
import os
import random
from logging import getLogger
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    ToPILImage,
)

import sys
sys.path.append("src/libs/")

from mean_std import get_mean, get_std

logger = getLogger(__name__)

def _pil2cv(image):
    """PIL -> OpenCV"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: #モノクロ
        pass
    elif new_image.shape[2] == 3: #カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: #透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError
    return new_image

def _cv2pil(image):
    """OpenCV -> PIL"""
    new_image = image.copy()
    if new_image.ndim == 2: #モノクロ
        pass
    elif new_image.shape[2] == 3: #カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4: #透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError
    new_image = Image.fromarray(new_image)
    return new_image

def _random_gamma_correction(img, gamma_min=0.5, gamma_max=1.5):
    # ガンマ補正
    gamma = np.random.uniform(gamma_min, gamma_max)

    table = (np.arange(256) / 255) ** gamma * 255
    table = np.clip(table, 0, 255).astype(np.uint8)

    return cv2.LUT(img, table)

class RandomGammaCorrection:
    def __init__(self, gamma_min=0.5, gamma_max=1.5) -> None:
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    
    def __call__(self, img:Image.Image) -> Image.Image:
        img = _pil2cv(img)
        img = _random_gamma_correction(img, self.gamma_min, self.gamma_max)
        return _cv2pil(img)
    
class BaseTransform():
    """
    画像のリサイズ，色を標準化

    Attributes
    __________
    resize: int
    mean : (R, G, B)
    std : (R, G, B)
    """

    def __init__(self,resize=256):
        self.mean = get_mean()
        self.std = get_std()
        self.base_transform = Compose([
            Resize((resize, resize)),
            ToTensor(),
            Normalize(self.mean, self.std),
        ])

    def __call__(self, img):
        return self.base_transform(img)

    

