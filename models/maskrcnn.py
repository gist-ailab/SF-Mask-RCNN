from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from models import resnet
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import torch
import torchvision
from torch.nn import functional as F
import numpy as np

from .backbone import get_backbone_with_fpn
from torch.hub import load_state_dict_from_url



def forward(self, images, targets=None):
    for i in range(len(images)):
        image = images[i]
        target = targets[i] if targets is not None else targets
        if image.dim() != 3:
            raise ValueError("images is expected to be a list of 3d tensors "
                                "of shape [C, H, W], got {}".format(image.shape))
        # image = self.normalize(image)
        # image, target = self.resize(image, target)
        images[i] = image
        if targets is not None:
            targets[i] = target
    image_sizes = [img.shape[-2:] for img in images]
    images = self.batch_images(images)
    image_list = ImageList(images, image_sizes)
    return image_list, targets


def maskrcnn_resnet_fpn(input_modality, fusion_method, backbone_name, 
                        pretrained=False, progress=True, num_classes=2, 
                        pretrained_backbone=True, trainable_backbone_layers=3, **kwargs):
    
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = get_backbone_with_fpn(input_modality, fusion_method, 
                                    backbone_name, pretrained_backbone, 
                                    trainable_backbone_layers)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512)), 
                                        aspect_ratios=((0.25, 0.5, 1.0, 2.0)))
    model = MaskRCNN(backbone, 2, rpn_anchor_generator=anchor_generator, **kwargs)

    return model

def get_model_instance_segmentation(cfg):

    GeneralizedRCNNTransform.forward = forward

    # initialize with pretrained weights
    model = maskrcnn_resnet_fpn(input_modality=cfg["input_modality"], 
                                fusion_method=cfg["fusion_method"], 
                                backbone_name=cfg["backbone_name"], 
                                pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # do category-agnostic instance segmentation (object or background)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes=2)

    return model