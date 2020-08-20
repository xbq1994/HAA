# HAA
[ACM MM 2020] Black Re-ID: A Head-shoulder Descriptor for the Challenging
Problem of Person Re-Identification [paper](http://arxiv.org/abs/2008.08528)

### Update
2020-08-12: Update Code. 
2020-08-20: Update paper link. 

### Bibtex
If you find the code useful, please consider citing our paper:
```
@InProceedings{xu2020ACM,
author = {Boqiang, Xu and Lingxiao, He and Xingyu, Liao and Wu,Liu and Zhenan, Sun and Tao, Mei},
title = {Black Re-ID: A Head-shoulder Descriptor for the Challenging Problem of Person Re-Identification},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM '20)},
month = {October},
year = {2020}
}
```

### Preparation
* Dataset: Black re-ID ([BaiDuDisk](https://pan.baidu.com/s/1xXxh5662ouoe8AQwN6VolA) ```pwd:xubq```  please add the path of the Black re-ID dataset to DATASETS.DATASETS_ROOT in ```./projects/Black_reid/configs/Base-HAA.yml```) 
* Pre-trained STN Model ([BaiDuDisk](https://pan.baidu.com/s/1OH428mw8w11tZ8aShc5A1A) ```pwd:xubq``` please add the path of the STN model to DATASETS.STN_ROOT in ```./projects/Black_reid/configs/Base-HAA.yml```) 


### Train
1. `cd` to folder:
```
 cd projects/Black_reid
```
2. If you want to train with 1-GPU, run:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml"
```
   if you want to train with 4-GPU, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml"
```

### Evaluation
To evaluate a model's performance, use:
```
CUDA_VISIBLE_DEVICES=0 train_net.py --config-file= "configs/HAA_baseline_blackreid.yml" --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Contacts
If you have any question about the project, please feel free to contact me.

E-mail: boqiang.xu@cripac.ia.ac.cn

## ACKNOWLEDGEMENTS
The code was developed based on the ’fast-reid’ toolbox https://github.com/JDAI-CV/fast-reid.
