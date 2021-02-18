import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import imgviz
import random
from torchvision.transforms import functional as F


DATASET_PATH = '../data/train'
SAVE_PATH = '../data/samples'


def process_depth(depth, min_depth=3500, max_depth=8000, unnormalize=False):
    if unnormalize:
        depth = cv2.resize(depth, (640, 360))
        depth = (max_depth-min_depth)*np.asarray(depth) + min_depth # 0-1 to min_depth to max_depth
        depth[depth > max_depth] = max_depth    
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.uint8(depth*255)
    return depth

if __name__ == "__main__":

    file_names = sorted(os.listdir(os.path.join(DATASET_PATH + '/rgb')))
    random.shuffle(file_names)

    for file_name in file_names:
        file_name = file_name[:-4]
        rgb = cv2.imread(os.path.join(DATASET_PATH, 'rgb', file_name + '.png'))
        instance_mask = cv2.imread(os.path.join(DATASET_PATH, 'seg', file_name + '.png'))
        depth = process_depth(np.load(os.path.join(DATASET_PATH, 'depth', file_name + '.npy')).astype(np.float32), unnormalize=True)
        depth_perlin = process_depth(np.load(os.path.join(DATASET_PATH, 'depth_perlin', file_name + '.npy')).astype(np.float32))
        depth_sparse_perlin = process_depth(np.load(os.path.join(DATASET_PATH, 'depth_sparse_perlin', file_name + '.npy')).astype(np.float32))

        obj_ids = np.unique(instance_mask)[1:]
        mask = instance_mask[:, :, 0]
        masks = mask == obj_ids[:, None, None]

        instviz = imgviz.instances2rgb(
            image = rgb,
            masks = masks,
            labels = obj_ids,
            alpha = 0.7,
            line_width = 3,
            boundary_width = 3
        )

        plt.figure(dpi=500)

        plt.subplot(3, 3, 1)
        plt.title("rgb", size=8)
        plt.imshow(rgb)
        plt.axis("off")

        plt.subplot(3, 3, 2)
        plt.title("instance mask", size=8)
        plt.imshow(instance_mask)
        plt.axis("off")

        plt.subplot(3, 3, 3)
        plt.title("rgb + instance mask", size=8)
        plt.imshow(instviz)
        plt.axis("off")

        plt.subplot(3, 3, 4)
        plt.title("depth", size=8)
        plt.imshow(depth)
        plt.axis("off")
        
        plt.subplot(3, 3, 5)
        plt.title("depth + perlin", size=8)
        plt.imshow(depth_perlin)
        plt.axis("off")

        plt.subplot(3, 3, 6)
        plt.title("depth + perlin + sparse edge", size=8)
        plt.imshow(depth_sparse_perlin)
        plt.axis("off")

        img = imgviz.io.pyplot_to_numpy()
        h, w, _ = img.shape
        cv2.putText(img, "n: next image / q: quit / s: save", (int(0.35*w), int(0.5*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(SAVE_PATH + '/' + file_name + '.png', img)
        cv2.imwrite('test.png', img)
        
        while True:
            if cv2.waitKey(0) == ord('q'):
                exit()
            elif cv2.waitKey(0) == ord('n'):
                cv2.destroyAllWindows()
                plt.close()
                break
            elif cv2.waitKey(0) == ord('s'):
                cv2.imwrite(SAVE_PATH + '/' +file_name + '.png', img)





