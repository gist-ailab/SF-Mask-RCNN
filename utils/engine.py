import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from .coco_eval import CocoEvaluator
from .metric_logger import *


def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, summary):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None

    for idx, (images , targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] \
            + loss_dict['loss_mask'] + loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']

        curr_itr = idx + epoch*len(data_loader) + 1
        summary.add_scalar('Loss/train/loss_classifier', loss_dict['loss_classifier'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_box_reg', loss_dict['loss_box_reg'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_mask', loss_dict['loss_mask'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_objectness', loss_dict['loss_objectness'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'].item(), curr_itr)
        summary.add_scalar('Loss/train/loss_total', losses, curr_itr)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(coco, model, data_loader, device, summary, epoch):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        
        p=coco_eval.stats

        summary.add_scalar('{}/AP'.format(iou_type), p[0], epoch)
        summary.add_scalar('{}/AP_50'.format(iou_type), p[1], epoch)
        summary.add_scalar('{}/AP_75'.format(iou_type), p[2], epoch)
        summary.add_scalar('{}/AP_S'.format(iou_type), p[3], epoch)
        summary.add_scalar('{}/AP_M'.format(iou_type), p[4], epoch)
        summary.add_scalar('{}/AP_L'.format(iou_type), p[5], epoch)

        summary.add_scalar('{}/AR_maxDets=1'.format(iou_type), p[6], epoch)
        summary.add_scalar('{}/AR_maxDets=10'.format(iou_type), p[7], epoch)
        summary.add_scalar('{}/AR_maxDets=100'.format(iou_type), p[8], epoch)
        summary.add_scalar('{}/AR_S_maxDets=100'.format(iou_type), p[9], epoch)
        summary.add_scalar('{}/AR_M_maxDets=100'.format(iou_type), p[10], epoch)
        summary.add_scalar('{}/AR_L_maxDets=100'.format(iou_type), p[11], epoch)        


    return coco_evaluator
