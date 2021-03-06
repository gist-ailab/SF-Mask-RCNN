from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
import loader.transforms as T
import cv2
from torchvision.transforms import functional as F


class RealDataset(Dataset):
    def __init__(self, dataset_path, mode="test", cfg=None):

        self.mode = mode # train mode or validation mode
        self.rgb_path = os.path.join(dataset_path, self.mode, "rgb")
        depth_folder = "depth_raw" if cfg["input_modality"] in "raw" else "depth_inpainted"
        self.depth_path = os.path.join(dataset_path, self.mode, depth_folder)
        self.seg_path = os.path.join(dataset_path, self.mode, "seg")

        self.rgb_list = list(sorted(os.listdir(self.rgb_path)))
        self.depth_list = list(sorted(os.listdir(self.depth_path)))
        self.seg_list = list(sorted(os.listdir(self.seg_path)))

        self.input_modality = cfg["input_modality"]
        self.width = cfg["width"]
        self.height = cfg["height"]

        self.min_depth = 0.35 / 0.0010000000474974513
        self.max_depth = 0.80 / 0.0010000000474974513

        self.rgb_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                                    ])


    def __getitem__(self, idx):

        inputs = dict.fromkeys(["rgb", "depth", "val_mask"])
        if 'rgb' in self.input_modality:
            rgb = Image.open(os.path.join(self.rgb_path, self.rgb_list[idx])).convert("RGB")
            rgb = rgb.resize((self.width, self.height))
            inputs["rgb"] = self.rgb_transform(rgb)
        
        if 'depth' in self.input_modality:
            depth = np.load(os.path.join(self.depth_path, self.depth_list[idx])).astype(np.float32)
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth = torch.from_numpy(depth).unsqueeze(-1).permute(2, 0, 1)
            
            # create corresponding validity mask            
            val_mask = torch.ones([self.height, self.width])
            val_mask[np.where(depth[0] == 0.0)] = 0
            val_mask = val_mask.unsqueeze(0)
            
            # depth clip & normalization
            depth[depth < self.min_depth] = self.min_depth
            depth[depth > self.max_depth] = self.max_depth
            depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
            inputs["depth"] =  torch.repeat_interleave(depth, 3, 0)
            inputs["val_mask"] = val_mask
        
        if inputs["rgb"] is not None and inputs["depth"] is not None:
            img = torch.cat((inputs["rgb"], inputs["depth"]), 0)
        elif inputs["rgb"] is not None:
            img = inputs["rgb"]
        elif inputs["depth"] is not None:
            img = inputs["depth"]
        if inputs["val_mask"] is not None:
            img = torch.cat((img, inputs["val_mask"]), 0)


        seg_mask_path = os.path.join(self.seg_path, self.seg_list[idx])
        seg_mask = Image.open(seg_mask_path).convert("L")
        seg_mask = seg_mask.resize((self.width, self.height), Image.NEAREST)
        seg_mask = np.array(seg_mask)
        # instances are encoded as different colors
        obj_ids = np.unique(seg_mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        seg_masks = seg_mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        temp_obj_ids = []
        temp_masks = []
        boxes = []
        for i in range(num_objs):
            pos = np.where(seg_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if int(xmax-xmin) < 1 or int(ymax-ymin) < 1 :
                continue
            temp_masks.append(seg_masks[i])
            temp_obj_ids.append(obj_ids[i])
            boxes.append([xmin, ymin, xmax, ymax])

        obj_ids = temp_obj_ids
        seg_masks = np.asarray(temp_masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = []
        for obj_id in obj_ids:
            if 1 <= obj_id:
                labels.append(1) 
            else:
                print("miss value error")
                exit(0)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        seg_masks = torch.as_tensor(seg_masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = seg_masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img, target

    def __len__(self):
        return len(self.rgb_list)