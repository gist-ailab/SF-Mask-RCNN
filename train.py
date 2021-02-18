import os
import yaml
import pathlib
import pprint
import argparse
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
from utils.coco_utils import get_coco_api_from_dataset


if __name__ == '__main__':

    # load arguments and cfgurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb', help="file name of configuration file")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--save_interval", type=int, default=1)
    args = parser.parse_args()
    with open('cfgs/' + args.cfg + '.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

    # fix seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    torch.manual_seed(7777)

    # load dataset
    if cfg["dataset"] == 'synthetic':
        train_dataset = SyntheticDataset(dataset_path=cfg["dataset_path"], mode="train", cfg=cfg)
    elif cfg["dataset"] == 'wisdom':
        train_dataset = WISDOMDataset(dataset_path=cfg["dataset_path"], mode="train", cfg=cfg)
    else:
        raise ValueError("Unsupported dataset type {} in your config file {}".format(cfg["dataset"], args.cfg))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg["batch_size"], 
                                               num_workers=4, shuffle=True, collate_fn=collate_fn)
   
    # logging
    now = datetime.datetime.now()
    logging_folder = os.path.join(pathlib.Path(__file__).parent.absolute(), 'logs', args.cfg )
    os.makedirs(logging_folder, exist_ok=True)
    summary = SummaryWriter(logdir=logging_folder)
    visualizer.draw_sample_images(train_dataset, logging_folder, cfg, "train")

    # load model
    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, args.gpu)
    model.to(device)

    start_epoch = 0
    if args.resume:
        if cfg["resume_ckp"] is None or "resume_ckp" not in cfg:
            # resume training from most recent epoch
            weights = os.listdir(logging_folder)
            start_epoch = sorted([int(w[:-4]) for w in weights if w[-4:] == ".tar"])[-1]
            print("Resume training from epoch", start_epoch)   
            model.load_state_dict(torch.load(logging_folder + "/" + str(start_epoch) + ".tar"))
        else:
            # resume training from checkpoint in config
            print("Resume training with pre-trained model", cfg["resume_ckp"])
            model.load_state_dict(torch.load(cfg["resume_ckp"]))
    
    if start_epoch >= cfg["max_epoch"]-1:
        print("Exit ==> start_epoch:", start_epoch, "max_epoch:", cfg["max_epoch"])
        exit()
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1, verbose=True)

    print("Start training")
    for epoch in range(start_epoch, cfg["max_epoch"]):
        # train_one_epoch(model, optimizer, train_loader, device, epoch, 100, summary)
        train_one_epoch(model, optimizer, train_loader, device, epoch, 3, summary)
        lr_scheduler.step()
        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), '{}/{}.tar'.format(logging_folder, epoch))
