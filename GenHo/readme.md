## Learning Few-Shot Generative Networks for Cross-Domain Data
### Official implementation of the ICCV 2019 paper
![Github](https://img.shields.io/badge/PyTorch-v0.4.1-red.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.5-green.svg?style=for-the-badge&logo=python)

![](https://github.com/SunnerLi/Few-Shot-GAN/blob/master/img/GenHo.png)

Abstract
---
This folder contains the code of the GenHo implementation. 

Usage
---
```
Training
# 1st step
$ python3 GenHo_step1.py --src_path StyleShoes

# 2nd step
$ python3 GenHo_step2.py --src_path StyleShoes \
--src_pair StyleShoes_100 \
--tar_pair RealShoes_100 \
--step1_path 'Save_model/Step1/step1_100.pth'

# 3rd step
$ python3 GenHo_step3.py --tar_path RealShoes_100 \
--step1_path 'Save_model/Step1/step1_100.pth'
--step2_path 'Save_model/Step2/step2_100.pth'

# 4th step
$ python3 GenHo_step4.py --src_pair StyleShoes_100 \
--tar_pair RealShoes_100 \
--step1_path 'Save_model/Step1/step1_100.pth'
--step2_path 'Save_model/Step2/step2_100.pth'

# 5th step
$ python3 GenHo_step5.py --step1_path 'Save_model/Step1/step1_100.pth'
--step3_path 'Save_model/Step3/step3_100.pth'
--prior_path 'Save_model/Step4/10000z.npz'
```

```
Inference
$ python3 GenHo_inference.py --step5_path 'Save_model/Step5/step5_100.pth'
```

Paper
---
* Hsuan-Kai Kao, Cheng-Che Lee, Hung-Yu Chen, Chia-Ming Cheng, and Wei-Chen Chiu, "Learning Few-Shot Generative Networks for Cross-Domain Data," ICCV 2019. 
