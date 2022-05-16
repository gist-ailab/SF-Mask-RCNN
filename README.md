# Synthetic RGB-D Fusion (SF) Mask R-CNN

Synthetic RGB-D Fusion (SF) Mask R-CNN for unseen object instance segmentation

> S. Back, J. Kim, R. Kang, S. Choi and K. Lee. **Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data.** 2020 IEEE International Conference on Image Processing (ICIP). IEEE, 2020. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9190804) [[Video]](https://youtu.be/eB4LKfYZwYo)

![Demo](./imgs/demo.gif)



## SF Mask R-CNN
---
Unseen object instance segmentation performance on [WISDOM](https://sites.google.com/view/wisdom-dataset/dataset_links) dataset 

|               Method              |   Input   | Use Synthetic Data |    Backbone   | mask AP | box AP |     Reference     |
|:---------------------------------:|:---------:|:------------------:|:-------------:|:-------:|:------:|:-----------------:|
|           SD Mask R-CNN           |   Depth   |     Yes (WISDOM)   | ResNet-35-FPN |   51.6  |    -   | [Danielczuk et al.](https://ieeexplore.ieee.org/abstract/document/8793744) |
|             Mask R-CNN            |    RGB    |         No         | ResNet-35-FPN |   38.4  |    -   | Danielczuk et al. |
|             Mask R-CNN            |    RGB    |         No         | ResNet-50-FPN |   40.1  |  36.7  |     [Ito et al.](https://openaccess.thecvf.com/content/ACCV2020/html/Ito_Point_Proposal_based_Instance_Segmentation_with_Rectangular_Masks_for_Robot_ACCV_2020_paper.html)    |
|               D-SOLO              |    RGB    |         No         | ResNet-50-FPN |   42.0  |  39.1  |     Ito et al.    |
|               PPIS                |    RGB    |         No         | ResNet-50-FPN |   52.3  |  48.1  |     Ito et al.    |
|             Mask R-CNN            |    RGB    |         Yes (Ours) | ResNet-50-FPN |   59.0  |  61.4  |     Ours          |
|             Mask R-CNN            |    Depth  |         Yes (Ours) | ResNet-50-FPN |   59.6  |  60.4  |     Ours          |
|    SF Mask R-CNN (early fusion)   | RGB-Depth |      Yes (Ours)    | ResNet-50-FPN |   55.5  |  57.2  |        [Ours](https://ieeexplore.ieee.org/abstract/document/9190804)       |
|    SF Mask R-CNN (late fusion)    | RGB-Depth |      Yes (Ours)    | ResNet-50-FPN |   58.7  |  59.0  |        Ours       |
|**SF Mask R-CNN (confidence fusion)**| RGB-Depth |      Yes (Ours)    | ResNet-50-FPN | **60.5**|**61.0**|        Ours       |


<img src="./imgs/model.png" width="800">
 

**SF Mask R-CNN** is an upgraded version of RGB-D fusion Mask R-CNN with a confidence map estimator [1].
The main differences from [1] are
- SF Mask R-CNN generates a self-attention map from RGB and inpainted depth (validity mask and raw depth were used in [1]) 
- This self-attention map is used as a confidence map; Thus, RGB and depth feature maps fused with spatial self-attention in four different scales.
- It was fined-tuned on WISDOM-REAL-Train (100 images) and evaluated on public unseen object instance segmentation dataset, WISDOM (The only custom industrial dataset was used previously)




## Updates
- SF Mask R-CNN has been released (2020/02/18)
- Train dataset has been released (2022/05/16)



## Getting Started
---
### Environment Setup

1. Setup anaconda environment
```
$ conda create -n sfmaskrcnn python=3.7
$ conda activate sfmaskrcnn
$ pip install torch torchvision
$ pip install imgviz tqdm tensorboardX pandas opencv-python imutils pyfastnoisesimd scikit-image pycocotools
$ pip install pyrealsense2 # for demo
$ conda activate sfmaskrcnn
```

2. Download the provided SF Mask R-CNN weights pre-trained on our custom dataset. 
- [rgb only](https://drive.google.com/file/d/128PI9Z6h3VBjBOVEIHV6lPUnk9YerfL6/view?usp=sharing)
- [depth only](https://drive.google.com/file/d/1UTnaehYIHbZzvGbHAAOqEjNxt9FN41RV/view?usp=sharing)
- [early fusion](https://drive.google.com/file/d/1H5HUo2qP56PfD893mqvuamiiyKfaT-Ux/view?usp=sharing)
- [late fusion](https://drive.google.com/file/d/1__Vqj9k8K8BckJaikrMqhYNdcjKuvVOf/view?usp=sharing)
- [confidence fusion](https://drive.google.com/file/d/11RonCOyv5YOBTZJXZxP75XT_V0xuEXTm/view?usp=sharing)

3. Download the WISDOM-Real dataset [[Link]](https://sites.google.com/view/wisdom-dataset/dataset_links)

4. Set the path to the dataset and pretrained weights (You can put this into your bash profile)
```
$ export WISDOM_PATH={/path/to/the/wisdom-real/high-res/dataset}
$ export WEIGHT_PATH={/path/to/the/pretrained/weights}

```


### Train

1. Download the synthetic train dataset at [GDrive](https://drive.google.com/file/d/1GN3_npwor1Yeng2n7XIqvzJ9BVSMoJ3W/view?usp=sharing)
2. Unzip the downloaded dataset, and modify the `dataset_path` of the config file correspondingly.

3. To train an SF Mask R-CNN (confidence fusion, RGB-noisy depth as input) on a synthetic dataset. 
```
$ python train.py --gpu 0 --cfg rgb_noisydepth_confidencefusion
```
4. To fine-tune the SF Mask R-CNN on WISDOM dataset
```
$ python train.py --gpu 0 --cfg rgb_noisydepth_confidencefusion_FT --resume
```

### Evaluation

To evaluate an SF Mask R-CNN (confidence fusion, RGB-noisy depth as input) on a WISDOM dataset
```
$ python eval.py --gpu 0 --cfg rgb_noisydepth_confidencefusion \
    --eval_data wisdom \
    --dataset_path $WISDOM_PATH \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 
```


### Visualization

To visualize the inference results of SF Mask R-CNN on a WISDOM dataset
```
$ python inference.py --gpu 0 --cfg rgb_noisydepth_confidencefusion \
    --eval_data wisdom --vis_depth \
    --dataset_path $WISDOM_PATH \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 
```

<img src="./imgs/example1.png" height="150">
<img src="./imgs/example2.png" height="150">
<img src="./imgs/example3.png" height="150">

Our custom synthetic dataset
```
$ python inference.py --gpu 0 --cfg rgb_noisydepth_confidencefusion \
    --eval_data synthetic --vis_depth \
    --dataset_path examples \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 
```
<img src="./imgs/syn_example.png" height="150">

### Demo with RealSense

To run real-time demo with realsense-d435
```
# SF Mask R-CNN (confidence fusion)
$ python demo.py --cfg rgb_noisydepth_confidencefusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_ConfidenceFusion.tar 

# SF Mask R-CNN (early fusion)
$ python demo.py --cfg rgb_noisydepth_earlyfusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_EarlyFusion.tar 


# SF Mask R-CNN (late fusion)
$ python demo.py --cfg rgb_noisydepth_latefusion \
    --weight_path $WEIGHT_PATH/SFMaskRCNN_LateFusion.tar 
```

## Authors
* **Seunghyeok Back** [seungback](https://github.com/SeungBack)
* **Raeyoung Kang** [raeyo](https://github.com/raeyo)
* **Taewon Kim** [ailabktw](https://github.com/ailabktw)
* **Joosoon Lee** [joosoon](https://github.com/joosoon)


## Citation
If you use our work in a research project, please cite our work:
```
[1] @inproceedings{back2020segmenting,
  title={Segmenting unseen industrial components in a heavy clutter using rgb-d fusion and synthetic data},
  author={Back, Seunghyeok and Kim, Jongwon and Kang, Raeyoung and Choi, Seungjun and Lee, Kyoobin},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={828--832},
  year={2020},
  organization={IEEE}
}
```
