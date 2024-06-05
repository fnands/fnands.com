---
layout: post
title: Reading Notes - OpenStereo
description: Some reading notes on the paper titled *OpenStereo - A Comprehensive Benchmark for Stereo Matching and Strong Baseline*
categories: [stereo, reading notes]
date: "2024-06-05"
author: "Ferdinand Schenck"
---

These are my notes on reading [OpenStereo: A Comprehensive Benchmark for
Stereo Matching and Strong Baseline](https://arxiv.org/abs/2312.00343) by Xianda Guo, Juntao Lu, Chenming Zhang, Yiqi Wang, Yiqun Duan, Tian Yang, Zheng Zhu, and Long Chen.


## Overview

In the paper they make the (in my experience well founded) claim that the evaluation methods for stereo matching algorithms are all over the place, making it difficult to make apples-to-apples comparisons and to judge the generalization ability of some stereo matching methods. Due to differences in training regimes and augmentation strategies, as well as inconsistent ablation strategies it can be hard to disentangle whether an improvement is due to architectural changes or training methodology.  

The paper introduces the [OpenStereo](https://github.com/XiandaGuo/OpenStereo) framework for comparing different techniques, as well as a model architecture called StereoBase, which is effectively a model put together from the pieces investigated in the study, which at the time of writing ranks in first place on the [KITTI2015 leaderboard](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) as well as reportedly achieving a new SOTA on the [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) test set. 

## Related work

The paper starts with an overview of deep-learning based stereo matching methods and groups them into two broad catagories:  

* Cost volume matching with 3D convolutions (CVM-Conv3D)
* Encoder-decoder with 2D convolutions (ED-Conv2D)

Broadly CVM-Conv3D outperform ED-Conv2D methods, although at the cost of much greater memory and compute requirements, which could prevent them from being used in real-time scenarios. 

## OpenStereo

The OpenStereo framework is introduced and consists of three main parts:   

1. A *Data module* which loads and preprocesses the datasets.
2. A *Modeling module* which defines a `BaseModel` which can be used to construct the specific network architectures. 
3. An *Evaluation module* which standardizes the evaluation methodology. 


Currently six datasets are supported: SceneFlow, KITTI2012, KITTI2015, ETH3D, Middlebury, and DrivingStereo. 

With the above tools a large amount of models can be put together and compared against each other. 

## Revisit Deep Stereo Matching

Eleven models are reconstructed using the OpenStereo framework and it is found that the OpenStereo implementations outperform the reference implementations when compared on the SceneFlow dataset. For the KITTI2015 dataset one metric is lower than the reference implementation.   


An ablation study is performed and it is found that:   

* Data augmentation: 
    - Most (standard) data augmentation techniques do more harm then good. 
    - Random crop, color augmentation and random erase deliver significant improvements however. 
* Feature extraction: 
    - Larger backbones do better
    - Using a pre-trained backbone leads to very significant improvements. 
* Cost construction:
    - Different cost-volume construction strategies are compared
    - 4D Methods (Height x Width x Depth x Channels) perform best
    - More channels correlates to better performance at a greater computational cost, but with diminishing returns. 
    - Optimal quality compute tradeoff seems to be the combined volume G8-G16 (from [Group-wise Correlation Stereo Network](https://arxiv.org/pdf/1903.04025))
* Disparity regression and refinement:
    - Larger and more computationally intensive refinement modules lead to better results
    - Best result achieved with ConvGRU from [RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/pdf/2109.07547) at a computational cost of ~20x as much as as the next best result for a ~30% improvement in EPE.

## A Strong Pipeline: StereoBase

By combining the learning of the above ablation study a new baseline architecture, called StereoBase was created, which achieves SOTA results on KITTI2012, KITTI2012 and SceneFlow, with competitive results on Driving Stereo. 

When trained on only SceneFlow (a synthetic dataset) it shows strong generalization to the KITTI2012, KITTI2015, Middlebury and ETH3D datasets, beating the other implemented architectures. 

## Questions

* How will stereo methods that do not conform to the ED-Conv2D or CVM-Conv3D catagories be accommodated in OpenStereo? 
* I think some of the augmentations might need to be re-thought:
    - As the authors mention, pixel alignment must be preserved
    - When flipping, you can't just naively flip the target, it needs to be inverted, which is a non-trivial operation. 
* Does it make sense to try even larger pre-trained backbones? 
    - The authors only test four different backbones. 
    - The largest is MobilenetV2 120d at 5.21M parameters. 
    - Does going even bigger do better? 
* Given this framework, could a neural architecture search be performed over an even larger set of backbones and components to find an even better architecture? 
* Does it make sense to also create a public leaderboard? 
    - Might be hard with the KITTI datasets that have their own board

## Key Takeways

The authors to a great job of categorizing recent developments in deep stereo matching and putting them on an equal footing.   

The OpenStereo framework is ambitious, and should really help ensure that future developments are rigorously tested and compared to what came before. 

The baseline meta-architecture they put together (OpenBase) shows that there is a lot that can be achieved by just doing a comprehensive ablation study, and should serve as a solid benchmark for future studies. 

