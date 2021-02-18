import torch
import torch.nn as nn
import datetime
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import argparse
from tqdm import tqdm
import os
from utils.engine import evaluate
import torchvision
import utils
from tensorboardX import SummaryWriter
import numpy as np
from models import maskrcnn
from loader import RealDataset
from utils.coco_utils import get_coco_api_from_dataset
from os import path
import glob
import json
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import torchvision.transforms as transforms
import threading
import yaml
import pathlib
import cv2
import random


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RealSenseD415:
    def __init__(self, depth_type):
        super().__init__()

        # settings for input
        self.rgb, self.depth, self.val_mask = None, None, None
        self.depth_type = depth_type
        self.min_dist = 0.35 / 0.0010000000474974513
        self.max_dist = 0.80 / 0.0010000000474974513

        # settings for realsense
        self.pipeline = rs.pipeline()
        self.rr_config = rs.config() # fix it to 1280, 720 on USB 3.0
        self.rr_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.rr_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        profile = self.pipeline.start(self.rr_config)

        depth_sensor = profile.get_device().first_depth_sensor()
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            print('%02d: %s'%(i, visulpreset))
            if visulpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        align_to = rs.stream.color # rs.align allows us to perform alignment of depth frames to others frames
        self.align = rs.align(align_to) # The "align_to" is the stream type to which we plan to align depth frames.
        # nearest_from_around - -Use the value from the neighboring pixel closest to the sensor
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter(mode=2)

    def start_stream(self):
        
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            raw_depth = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            rgb = aligned_frames.get_color_frame()

            if rgb and raw_depth:
                self.rgb, self.depth, self.val_mask = self.preprocess(rgb, raw_depth)


    def preprocess(self, rgb, raw_depth):
        # fill raw depth with filter
        filled_depth = self.spatial_filter.process(raw_depth)
        filled_depth = self.hole_filling_filter.process(filled_depth)
        # rs data -> numpy
        rgb = np.asarray(rgb.get_data())
        rgb = cv2.resize(rgb, (640, 360))
        if self.depth_type == 'raw':
            depth = np.asanyarray(raw_depth.get_data())
        elif self.depth_type == 'inpainted':
            depth = np.asanyarray(filled_depth.get_data())
        # numpy -> torch
        depth = np.expand_dims(np.asarray(depth), -1)
        depth = np.repeat(depth, 3, -1)
        depth = cv2.resize(depth, (640, 360))
        depth = torch.from_numpy(np.float32(depth))
        # 3500-8000 to 0-1
        depth[self.min_dist > depth] = self.min_dist
        depth[self.max_dist < depth] = self.max_dist
        depth = (depth-self.min_dist) / (self.max_dist-self.min_dist) 
        depth = depth.transpose(0, 2).transpose(1, 2)
        # depth -> val_mask
        val_mask = torch.ones([depth.shape[1], depth.shape[2]])
        val_mask[np.where(depth[0] == 0.0)] = 0
        val_mask = val_mask.unsqueeze(0)
    
        return rgb, depth, val_mask
    
    def stop(self):
        # Stop streaming
        self.pipeline.stop()



def draw_prediction(rgb_img, object_idx, img_arr, boxes, score, thresh):

    if len(object_idx) == 0:
        return False

    rgb_img = np.uint8(rgb_img.transpose(0, 1, 2))

    for i in object_idx:
        if score[i] > thresh:
            c = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            color = np.uint8(c)
            obj_img = img_arr[i]

            obj_img[obj_img >= 0.5] = 1
            obj_img[obj_img < 0.5] = 0

            r = obj_img * color[0]
            g = obj_img * color[1]
            b = obj_img * color[2]

            stacked_img = np.stack((r, g, b), axis=0)
            stacked_img = stacked_img.transpose(1, 2, 0)

            rgb_img = cv2.addWeighted(rgb_img, 1, stacked_img.astype(np.uint8), 0.5, 0)
            cv2.rectangle(rgb_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), c, 2)
            cv2.putText(rgb_img, str(score[i].item())[:4], (boxes[i][0], int(boxes[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)

    rgb_img = np.uint8(rgb_img)

    return rgb_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='rgb_raw_depth_confidencefusion', help=".yaml")
    parser.add_argument("--thresh", type=float, default=0.7, help="thresh")
    parser.add_argument("--weight_path", type=str, default=None, help="if it is given, evaluate this")
    args = parser.parse_args()

    with open('cfgs/' + args.cfg + '.yaml') as config_file:
        cfg = yaml.safe_load(config_file)

    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    model.load_state_dict(torch.load(args.weight_path))
    model.cuda()
    model.eval()


    if "raw" in cfg["input_modality"]:
        rs415 = RealSenseD415(depth_type='raw')
        print("depth type: raw")
    else:
        rs415 = RealSenseD415(depth_type='inpainted')
        print("depth type: inpatined")

    cam_thread = threading.Thread(target=rs415.start_stream, args=())
    cam_thread.daemon = True
    cam_thread.start()

    rgb_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                                        ])

    while True:
        if rs415.rgb is not None:
            rgb = rs415.rgb
            if cfg["input_modality"] == "rgb": 
                img = rgb_transform(rgb)
            elif cfg["input_modality"] == "inpainted_depth" or cfg["input_modality"] == "raw_depth": 
                img = rs415.depth
            elif cfg["input_modality"] == "rgb_inpainted_depth" or cfg["input_modality"] == "rgb_raw_depth": 
                img = torch.cat([rgb_transform(rgb), rs415.depth, rs415.val_mask], dim=0)             
        else:
            continue

        img = img.unsqueeze(0)
        images = list(image.to(device) for image in img)

        pred_result = model(images)[0]
        pred_mask = pred_result['masks'].cpu().detach().numpy()
        pred_labels = pred_result['labels']
        pred_scores = pred_result['scores']
        pred_boxes = pred_result['boxes']
        img_arr = []

        for num in range(len(pred_result['labels'])):
            mask_arr = pred_mask[num][0]
            img_arr.append(mask_arr)

        object_idx = [i for i in range(len(pred_labels)) if pred_labels[i] == 1]
        result_image = draw_prediction(rgb, object_idx, img_arr, pred_boxes, pred_scores, args.thresh)
        cv2.imshow("demo", result_image)
        k = cv2.waitKey(100)
        if k == 27: # esc key
            cv2.destroyAllWindow()
            rs415.stop()
            break

