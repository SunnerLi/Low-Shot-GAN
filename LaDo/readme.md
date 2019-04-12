## Learning Few-Shot Generative Networks for Cross-Domain Data
### Official implementation of the ICCV 2019 paper
![Github](https://img.shields.io/badge/PyTorch-v0.4.1-red.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.5-green.svg?style=for-the-badge&logo=python)

![](https://github.com/SunnerLi/Few-Shot-GAN/blob/master/img/LaDo_structure.png)

Abstract
---
This folder records the code of LaDo approach. The meaning of the each folder or file will be listed in below.
* `torchvision_sunner`: unlike the original [one](https://github.com/SunnerLi/Torchvision_sunner), we modify a little and use in our work.
* `lib`: This folder includes the whole relative scripts which will be used in the LaDo. 

Usage
---
Train for LaDo! **You should assign the essential parameters** which are described in the comment[[1](https://github.com/SunnerLi/Few-Shot-GAN-LaDo/blob/master/LaDo_train_1st_step.py#L30-L45), [2](https://github.com/SunnerLi/Few-Shot-GAN-LaDo/blob/master/LaDo_train_2nd_step.py#L27-L42)]
```
# 1st stage
$ python3 LaDo_train_1st_step.py --src_image_folder StyleShoes \
--src_pair_image_folder StyleShoes_100 \
--tar_pair_image_folder RealShoes_100 \
--root_folder Training_result

# 2nd stage
$ python3 LaDo_train_2nd_step.py --model_path_1st Training_result \
--src_pair_image_folder StyleShoes_100 \
--tar_pair_image_folder RealShoes_100 \
--root_folder Training_result
```

Finally, you can do the inference by your own.
```
$ python3 LaDo_random_sample.py --model_path LaDo_result/models_2nd/00100.pth
```

Paper
---
* Shuian-Kai Kao, Cheng-Che Lee, Hung-Yu Chen, Chia-Ming Cheng, and Wei-Chen Chiu, "Learning Few-Shot Generative Networks for Cross-Domain Data," ICCV 2019. 
