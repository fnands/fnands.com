---
layout: post
title: Reading Notes: OpenStereo
description: Some reading notes on the paper titled: *OpenStereo: A Comprehensive Benchmark for Stereo Matching and Strong Baseline*
categories: [stereo, reading notes]
draft: true
date: "2099-05-04"
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

* 


## Questions

* How will stereo methods that do not conform to the ED-Conv2D or CVM-Conv3D catagories be accommodated in OpenStereo? 