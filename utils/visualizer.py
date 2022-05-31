import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
import imgviz
import matplotlib.pyplot as plt

def draw_sample_images(loader, save_path, cfg, prefix, n_images=10):
    for i in range(n_images):
        img, target = loader.__getitem__(i)
        images = dict.fromkeys(["rgb", "depth", "val_mask"])
        if cfg["input_modality"] == "rgb": 
            images["rgb"] = img[:3, :, :]
        elif cfg["input_modality"] == "inpainted_depth": 
            images["depth"] = img[:3, :, :]
        elif cfg["input_modality"] == "raw_depth":
            images["depth"] = img[:3, :, :]
            images["val_mask"] = img[3, :, :]
        elif cfg["input_modality"] == "rgb_inpainted_depth":
            images["rgb"] = img[:3, :, :]
            images["depth"] = img[3:6, :, :]
        elif cfg["input_modality"] == "rgb_raw_depth":
            images["rgb"] = img[:3, :, :]
            images["depth"] = img[3:6, :, :]
            images["val_mask"] = img[6, :, :]

        for k in images.keys():
            if images[k] is None: continue
            if k == "rgb":
                images[k] = F.normalize(images[k], 
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                        std=[1/0.229, 1/0.224, 1/0.225])
            img = F.to_pil_image(images[k])
            img.save("{}/{}_{}_{}.png".format(save_path, prefix, k, i))

def draw_prediction(image, pred, thresh, vis_depth=False):
    
    inv_normalize = transforms.Compose([
                        transforms.Normalize(
                            mean = [ 0., 0., 0. ],
                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                        transforms.Normalize(
                            mean = [ -0.485, -0.456, -0.406 ],
                            std = [ 1., 1., 1. ]),
                        ])
    rgb = image[:3]
    rgb = inv_normalize(rgb)
    rgb = rgb.transpose(0, 2).transpose(0, 1) * 255
    rgb = np.uint8(rgb)

    scores = pred["scores"].detach().numpy()
    boxes = pred["boxes"].detach().numpy()
    masks = pred["masks"].detach().numpy()
    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0
    cnd = scores[:] > thresh

    masks = np.array(np.squeeze(masks), dtype=np.bool)
    instviz = imgviz.instances2rgb(image=rgb, masks=masks[cnd, :, :], labels=list(range(len(scores[cnd]))), 
                                    captions=[str(round(x, 2)) for x in scores[cnd]])
    plt.figure(dpi=200)
    plt.imshow(instviz)
    plt.axis("off")
    instviz = imgviz.io.pyplot_to_numpy()
    instviz = cv2.cvtColor(cv2.resize(instviz, (rgb.shape[1], rgb.shape[0])), cv2.COLOR_BGR2RGB)
    instviz = np.hstack((rgb, instviz))

    if vis_depth:
        # add depth image
        depth = image[3:6]
        depth = depth.transpose(0, 2).transpose(0, 1) * 255
        depth = np.uint8(depth)
        instviz = np.hstack((instviz, depth))

    return instviz
