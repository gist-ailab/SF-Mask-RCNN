import argparse
import cv2
import os
import pathlib
import yaml
from tqdm import tqdm

import torch
from models import maskrcnn
from loader import SyntheticDataset, WISDOMDataset
from utils.visualizer import draw_prediction

if __name__ == '__main__':

    # load arguments and cfgurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="gpu number to use. 0, 1")
    parser.add_argument("--cfg", type=str, default='rgb', help="file name of configuration file")
    parser.add_argument("--eval_data", default='wisdom', choices=['synthetic', 'wisdom'])
    parser.add_argument("--dataset_path", type=str, help="path to dataset for evaluation")
    parser.add_argument("--num_sample", type=int, default=None, help="the number of data for drawing")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--vis_depth", action="store_true", help="draw RGB and depth together if true")
    parser.add_argument("--epoch_list", type=str, default=None, help="list of comma splited epochs to evaluate")
    parser.add_argument("--weight_path", type=str, default=None, help="if it is given, evaluate this")
    args = parser.parse_args()
    
    with open('cfgs/' + args.cfg + '.yaml') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    # fix seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True    
    torch.manual_seed(7777)

    # load model
    model = maskrcnn.get_model_instance_segmentation(cfg=cfg)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, args.gpu)
    model.to(device)

    # load dataset
    if args.eval_data == 'synthetic':
        val_dataset = SyntheticDataset(dataset_path=args.dataset_path, mode="train", cfg=cfg)
    elif args.eval_data == 'wisdom':
        val_dataset = WISDOMDataset(dataset_path=args.dataset_path, mode="val", cfg=cfg, num_sample=args.num_sample)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, num_workers=1, shuffle=False)

    # path to checkpoint files
    logging_folder = os.path.join(pathlib.Path(__file__).parent.absolute(), 'logs', args.cfg )
    save_dir = os.path.join(logging_folder, "vis_result_{}".format(args.eval_data))
    print("Images are saved at", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cpu_device = torch.device("cpu")

    if args.epoch_list is None:
        # evaluate all trained epochs
        weights = os.listdir(logging_folder)
        epoch_list = sorted([int(w[:-4]) for w in weights if w[-4:] == ".tar"])
    else:
        # evaluate only assigned epochs
        epoch_list = sorted([int(epoch) for epoch in args.epoch_list.split(',')])

    # inference
    if args.weight_path is not None:
        print("Inference with", args.weight_path)
        model.load_state_dict(torch.load(args.weight_path))
        model.eval()
        for img_idx, (image, _) in enumerate(tqdm(val_loader)):
            image = list(img.to(device) for img in image)
            outputs = model(image)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # draw result image
            vis_image = draw_prediction(image[0].to(cpu_device), outputs[0], args.thresh, args.vis_depth)
            # save image
            save_name = "img{:03d}.png".format(img_idx)
            cv2.imwrite(os.path.join(save_dir, save_name), vis_image)

    else:
        for epoch in epoch_list:
            model.load_state_dict(torch.load(logging_folder + "/" + str(epoch) + ".tar"))
            model.eval()
            for img_idx, (image, _) in enumerate(tqdm(val_loader)):
                image = list(img.to(device) for img in image)
                outputs = model(image)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                # draw result image
                vis_image = draw_prediction(image[0].to(cpu_device), outputs[0], args.thresh, args.vis_depth)
                # save image
                save_name = "img{:03d}_epoch_{}.png".format(img_idx, epoch)
                cv2.imwrite(os.path.join(save_dir, save_name), vis_image)
