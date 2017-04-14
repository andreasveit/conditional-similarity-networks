# Conditional Similarity Networks (CSNs)

This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) presented at CVPR 2017. 

The code is based on the [PyTorch example for training ResNet on Imagenet](https://github.com/pytorch/examples/tree/master/imagenet) and the [Triplet Network example](https://github.com/andreasveit/triplet-network-pytorch).

## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Citing](#citing)
0. [Contact](#contact)

## Introduction
What makes images similar? To measure the similarity between images, they are typically embedded in a feature-vector space, in which their distance preserve the relative dissimilarity. However, when learning such similarity embeddings the simplifying assumption is commonly made that images are only compared to one unique measure of similarity.

[Conditional Similarity Networks](https://arxiv.org/abs/1603.07810) address this shortcoming by learning a nonlinear embeddings that gracefully deals with multiple notions of similarity within a shared embedding. Different aspects of similarity are incorporated by assigning responsibility weights to each embedding dimension with respect to each aspect of similarity.

<img src="https://github.com/andreasveit/conditional-similarity-networks/blob/master/images/csn_overview.png?raw=true" width="600">

Images are passed through a convolutional network and projected into a nonlinear embedding such that different dimensions encode features for specific notions of similarity. Subsequent masks indicate which dimensions of the embedding are responsible for separate aspects of similarity. We can then compare objects according to various notions of similarity by selecting an appropriate masked subspace.

## Usage
The detault setting for this repo is a CSN with fixed masks, an embedding dimension 64 and four notions of similarity.

You can download the Zappos dataset as well as the training, validation and test triplets used in the paper with

```sh
python get_data.py
```

The network can be simply trained with `python main.py` or with optional arguments for different hyperparameters:
```sh
$ python main.py --name {your experiment name} --learned --num_traintriplets 200000
```

Training progress can be easily tracked with [visdom](https://github.com/facebookresearch/visdom) using the `--visdom` flag. It keeps track of the learning rate, loss, training and validation accuracy both for all triplets as well as separated for each notion of similarity, the embedding norm, mask norm as well as the masks.

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
