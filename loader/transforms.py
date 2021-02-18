import random
import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
from tqdm import tqdm
import glob
import argparse
import imutils
import pyfastnoisesimd as fns
import random
import math
from scipy import sparse



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, mask=None):
        for t in self.transforms:
            image, target = t(image, target, mask)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target, mask=None):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target, mask=None):
        image = F.to_tensor(image)
        if mask is not None:
            mask = F.to_tensor(mask)
            return image, target, mask
        else:
            return image, target

class AddNoiseSparse(object):
    def __call__(self, image, target, mask):
        image = 255*np.asarray(image)
        image = np.expand_dims(image, -1)

        image = perlinDistortion(image, image.shape[1], image.shape[0])
        # image = IRE(image)
        image = ORE(image, mask)
        # image = addSparsity(image)
        image = np.expand_dims(image, -1)

        image = np.repeat(image, 3, -1)

        return image, target

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def RandomErasing(image, mask, scale=(0.002, 0.005), ratio=(0.7, 1.3)):
    # Image-aware Random Erasing
    img_c, img_h, img_w = image.shape
    area = img_h * img_w
    for _ in range(random.randint(1,3)):
        erase_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])

        h = int(round(math.sqrt(erase_area * aspect_ratio)))
        w = int(round(math.sqrt(erase_area / aspect_ratio)))

        if h < img_h and w < img_w:
            i = random.randint(0, img_h - h)
            j = random.randint(0, img_w - w)
        v = 0

        image[:, i:i + h, j:j + w] = v
        mask[:, i:i + h, j:j + w] = v
    return image, mask

def SaltPepperNoise(image, val_mask):

    density1 = random.uniform(0.985, 0.995)
    mask1 = sparse.random(image.shape[1], image.shape[2], density1,
        format='csr', data_rvs=np.ones, dtype='f')
    mask1 = torch.Tensor(mask1.toarray()).unsqueeze(0)
    image = image*mask1 # randomly insert 0
    val_mask = val_mask*mask1

    density2 = random.uniform(0.985, 0.995)
    mask2 = sparse.random(image.shape[1], image.shape[2], density2,
        format='csr', data_rvs=np.ones, dtype='f')
    mask2 = torch.Tensor(mask2.toarray()).unsqueeze(0)
    # mask2 = mask2.repeat(3, 1, 1)
    image[mask2==0] = 1 # randomly insert 1
    val_mask[mask2==0] = 0

    return image, val_mask
