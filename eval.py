import os
import yaml
import pathlib
import pprint
import argparse
import glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import datetime
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from tensorboardX import SummaryWriter


from models import maskrcnn
from loader import SyntheticDataset, WISDOMDataset
from utils import visualizer
from utils.engine import train_one_epoch, evaluate, collate_fn
from utils.coco_utils import get_coco_api_from_dataset, coco_to_excel


if __name__ == '__main__':

    # load arguments and cfgurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb', help="file name of configuration file")
    parser.add_argument("--eval_data", default='wisdom', choices=['synthetic', 'wisdom'], help="test dataset for evaluation.")
    parser.add_argument("--dataset_path", type=str, help="path to the evaluation dataset")
    parser.add_argument("--epoch_list", type=str, default=None, help="list of epochs to evaluate e.g. 0,1,2")
    parser.add_argument("--weight_path", type=str, default=None, help="if it is given, evaluate this")
    parser.add_argument("--write_excel", action="store_true")
    args = parser.parse_args()
    with open('cfgs/' + args.cfg + '.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

    # fix seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    torch.manual_seed(7777)

    # logging
    now = datetime.datetime.now()
    logging_folder = os.path.join(pathlib.Path(__file__).parent.absolute(), 'logs', args.cfg )
    os.makedirs(logging_folder, exist_ok=True)
    summary = SummaryWriter(logdir=logging_folder)

    # load dataset
    if args.eval_data == 'synthetic':
        val_dataset = SyntheticDataset(dataset_path=args.dataset_path, mode="train", cfg=cfg)
    elif args.eval_data == 'wisdom':
        val_dataset = WISDOMDataset(dataset_path=args.dataset_path, mode="val", cfg=cfg)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=1,
                                            shuffle=False, collate_fn=collate_fn)
    print("Loading coco..")
    coco = get_coco_api_from_dataset(val_loader.dataset)

    # load model
    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, args.gpu)
    model.to(device)


    if args.weight_path is not None:
        print("Evaluating", args.weight_path)
        model.load_state_dict(torch.load(args.weight_path))
        coco_evaluator = evaluate(coco, model, val_loader, device=device, summary=summary, epoch=-1)

    else:
        if args.epoch_list is None:
            # evaluate all trained epochs
            weights = os.listdir(logging_folder)
            epoch_list = sorted([int(w[:-4]) for w in weights if w[-4:] == ".tar"])
        else:
            # evaluate only assigned epochs
            epoch_list = sorted([int(epoch) for epoch in args.epoch_list.split(',')])
        for epoch in epoch_list:
            print("Evaluating", epoch)
            model.load_state_dict(torch.load(logging_folder + "/" + str(epoch) + ".tar"))
            coco_evaluator = evaluate(coco, model, val_loader, device=device, summary=summary, epoch=epoch)
            if args.write_excel:
                coco_to_excel(coco_evaluator, epoch, logging_folder, args.eval_data)   
