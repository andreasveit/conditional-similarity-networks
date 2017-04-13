# Conditional Similarity Networks (CSNs)

This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) at CVPR 2017. The code is based on the [PyTorch example for training ResNet on Imagenet](https://github.com/pytorch/examples/tree/master/imagenet) and the [Triplet Network example](https://github.com/andreasveit/triplet-network-pytorch).

## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Citing](#citing)
0. [Contact](#contact)

## Introduction

## Usage
The detault setting for this repo is a CSN with fixed masks, an embedding dimension 64 and four notions of similarity.

You can download the Zappos dataset as well as the training, validation and test triplets used in the paper with

```sh
python get_data.py
```

Usage with example optional arguments for different hyperparameters:
```sh
$ python main.py --name {your experiment name} --learned --num_traintriplets 200000
```

You can easily track the training progres with [visdom](https://github.com/facebookresearch/visdom) using the `--visdom` flag. It keeps track of the learning rate, loss, training and validation accuracy both for all triplets as well as separated for each notion of similarity, the embedding norm, mask norm as well as the masks.

<img src="https://github.com/andreasveit/conditional-similarity-networks/blob/master/images/visdom.png?raw=true" width="500">

By default the training code keeps track of the model with the highest performance on the validation set. Thus, after the model has converged, it can be directly evaluated on the test set as follows
```sh
$ python main.py --test --resume runs/{your experiment name}/model_best.pth.tar
```

## Citing
If you find this helps your research, please consider citing:

```
@conference{Veit2017,
title = {Conditional Similarity Networks},
author = {Andreas Veit and Serge Belongie and Theofanis Karaletsos},
year = {2017},
journal = {Computer Vision and Pattern Recognition (CVPR)},
}
```

## Contact
andreas at cs dot cornell dot edu 

Any discussions, suggestions and questions are welcome!
