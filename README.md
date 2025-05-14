# Adaptive Prototype Replay for Class Incremental Semantic Segmentation

This is an official implementation of the paper "Adaptive Prototype Replay for Class Incremental Semantic Segmentation", accepted by AAAI 2025.
[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33188)

<img src = "https://github.com/zhu-gl-ux/Adapter/blob/main/figures/overview.png" width="100%" height="100%">

# Abstract
Class incremental semantic segmentation (CISS) aims to segment new classes during continual steps while preventing the forgetting of old knowledge. Existing methods alleviate catastrophic forgetting by replaying distributions of previously learned classes using stored prototypes or features. However, they overlook a critical issue: in CISS, the representation of class knowledge is updated continuously through incremental learning, whereas prototype replay methods maintain fixed prototypes. This mismatch between updated representation and fixed prototypes limits the effectiveness of the prototype replay strategy. To address this issue, we propose the Adaptive prototype replay (Adapter) for CISS in this paper. Adapter comprises an adaptive deviation compensation (ADC) strategy and an uncertainty-aware constraint (UAC) loss. Specifically, the ADC strategy dynamically updates the stored prototypes based on the estimated representation shift distance to match the updated representation of old class. The UAC loss reduces prediction uncertainty, aggregating discriminative features to aid in generating compact prototypes. Additionally, we introduce a compensation-based prototype similarity discriminative (CPD) loss to ensure adequate differentiation between similar prototypes, thereby enhancing the efficiency of the adaptive prototype replay strategy. Extensive experiments on Pascal VOC and ADE20K datasets demonstrate that Adapter achieves state-of-the-art results and proves effective across various CISS tasks, particularly in challenging multi-step scenarios.


# Getting Started

### Requirements
- python==3.11.4
- torch==1.12.1
- torchvision==0.13.1
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib


### Datasets
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
    --- ADEChallengeData2016
        --- annotations
            --- training
            --- validation
        --- images
            --- training
            --- validation
```
You can download the original dataset in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit). To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). For ADE20K, you can dawnload the dataset in [here](http://sceneparsing.csail.mit.edu/).

### Class-Incremental Segmentation Segmentation on VOC 2012

```Shell

GPU=0
BS=24 
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='1-1'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 



NAME='Adapter'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

### Class-Incremental Segmentation Segmentation on ADE20K

```shell
GPU=0,1
BS=12  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='100-50'
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0

NAME='Adapter'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```


### Testing

```Shell
python eval_voc.py --device 0 --test --resume path/to/weight.pth
```
### Qualitative Results

<img src = "https://github.com/zhu-gl-ux/Adapter/blob/main/figures/voc-comparison.png" width="100%" height="100%">
<p align="center">Figure 1:Qualitative results of Ours, PLOP, DKD, and STAR on VOC 15-1.</p>
<img src = "https://github.com/zhu-gl-ux/Adapter/blob/main/figures/voc-steps.png" width="100%" height="100%">
<p align="center">Figure 2:Qualitative results of Ours on VOC 15-1 6 steps.</p>
<img src = "https://github.com/zhu-gl-ux/Adapter/blob/main/figures/ade-steps.png" width="100%" height="100%">
<p align="center">Figure 2:Qualitative results of Ours on ADE 100-10 6 steps.</p>


## Citation
```
@inproceedings{zhu2025adaptive,
  title={Adaptive prototype replay for class incremental semantic segmentation},
  author={Zhu, Guilin and Wu, Dongyue and Gao, Changxin and Wang, Runmin and Yang, Weidong and Sang, Nong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={10},
  pages={10932--10940},
  year={2025}
}
```

## Acknowledgements
* This code is based on [DKD](https://github.com/cvlab-yonsei/DKD) ([2022-NeurIPS]) and [STAR](https://github.com/jinpeng0528/STAR) [2023-NeurIPS].
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
