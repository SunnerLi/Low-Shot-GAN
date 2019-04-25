## Learning Few-Shot Generative Networks for Cross-Domain Data
### Official implementation of the ICCV 2019 paper
![Github](https://img.shields.io/badge/PyTorch-v0.4.1-red.svg?style=for-the-badge&logo=data:image/png)
![Github](https://img.shields.io/badge/python-3.5-green.svg?style=for-the-badge&logo=python)

<p align="center">
  <img src="https://github.com/SunnerLi/Few-Shot-GAN/blob/master/img/teaser_figure.png" width=450 height=320/>
</p> 
    
> **Learning Few-Shot Generative Networks for Cross-Domain Data** <br/>
> Hsuan-Kai Kao (NCTU), Cheng-Che Lee (NCTU), Hung-Yu Chen (NCTU), Chia-Ming Cheng (MediaTek), and Wei-Chen Chiu (NCTU) <br/>
>
> **Abstract**  *In this paper we tackle a novel problem of learning generators for cross-domain data under a specific scenario of few-shot learning. Basically, given a source domain with sufficient amount of training data, we aim to
transfer the knowledge of its generative process to another target domain which not only has few data samples but also the domain shift with respect to the source domain. This problem is of great potential in practical use as the large data consumption for learning targetdomain generator can be alleviated. Built upon a crossdomain dataset that (1) each of the few shots in the target domain has its correspondence in the source and (2) these two domains share the similar content information but different appearance, two approaches are proposed: a Latent-Disentanglement-Orientated model (LaDo) and a
Generative-Hierarchy-Oriented (GenHo) model. Our LaDo and GenHo approaches address the problem from different perspective, where the former relies on learning the disentangled representation composed of domain-invariant content features and domain-specific appearance ones; while the later decomposes the generative process of a generator into two parts, which are related to content and appearance synthesis respectively. We perform extensive experiments under various settings of cross-domain data and demonstrate the efficacy of our proposed models for generating target-domain data with abundant content variance as in the source domain, which leads to the favourable performance in comparison to several baselines.*


Usage
---
Please refer to the individual folder to check for more detail.

Citation
---
If you utilize our paper idea to extend in your research, please cite our paper:
```
@inproceedings{kao2018few,
  title={Learning Few-Shot Generative Networks for Cross-Domain Data},
  author={Kao, Hsuan-Kai and Lee, Cheng-Che and Chen, Hung-Yu and Cheng, Chia-Ming and Chiu, Wei-Chen},
  booktitle={ICCV},
  year={2019}
}
```
