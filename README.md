![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

# 深度学习模型

本项目讲述了深度学习中的结构、模型和技巧，使用的深度学习框架是 TensorFlow 和 PyTorch，代码和图文都以 Jupyter Notebook 的形式编写。

## 传统机器学习

- 感知器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/perceptron.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/perceptron.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/basic-ml/perceptron.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/perceptron.ipynb)]
- 逻辑回归（二分类器） [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/logistic-regression.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/logistic-regression.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/basic-ml/logistic-regression.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/logistic-regression.ipynb)]
- Softmax 回归（多分类器） [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/softmax-regression.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/softmax-regression.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/basic-ml/softmax-regression.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/softmax-regression.ipynb)]

## 多层感知器

- 多层感知器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-basic.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb)]
- 带有 Dropout 的多层感知器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-dropout.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-dropout.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-dropout.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-dropout.ipynb)]
- 带有 Batch Normalization 的多层感知器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-batchnorm.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-batchnorm.ipynb)]
- 手写反向传播的多层感知器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb)]

## 卷积神经网络

#### 基本

- 卷积神经网络 [TensorFlow 1: [GitHub](tensorflow1_ipynb/cnn/cnn-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/cnn/cnn-basic.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-basic.ipynb)]
- 使用 He 初始化的卷积神经网络  [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-he-init.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-he-init.ipynb)]

#### 概念

- 使用卷积层等效替换全连接层 [PyTorch: [GitHub](pytorch_ipynb/cnn/fc-to-conv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/fc-to-conv.ipynb)]

#### 全卷积

- 全卷积网络 [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-allconv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-allconv.ipynb)]

#### 数据集介绍

| 数据集 | 中文名称 | 样本数 | 图像尺寸 | 官方网站 |
| ---- | ---- | ---- | ---- | ---- |
| MNIST | 手写数字数据集 | 训练集 60000，测试集 10000 | (28, 28) | [MNIST](http://yann.lecun.com/exdb/mnist/) | 
| CIFAR-10 | 加拿大高等研究院-10 | 训练集 50000，测试集 10000 | (32, 32) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | 
| SVHN | 街景门牌号 | 训练集 73257，测试集 26032，额外 531131 | 尺寸不一，裁剪后 (32, 32) | [SVHN](http://ufldl.stanford.edu/housenumbers/) |
| CelebA | 名人面部属性数据集 | 202599 | 尺寸不一，图像宽度超过 200 | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| Quickdraw | 快速涂鸦数据集 | 5000 万 | 原始尺寸是 (256, 256)，裁剪后为 (32, 32) | [Quickdraw](https://github.com/googlecreativelab/quickdraw-dataset) |

#### 模型搭建与训练

| 数据集 | 模型 | 任务 | 地址 | 测试集准确率 |
| ---- | ---- | ---- | ---- | ---- |
| CIFAR-10 | LeNet-5 | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb) | 61.70% |
| CIFAR-10 | Network in Network | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/nin-cifar10.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10.ipynb) | 70.67% |
| CIFAR-10 | AlexNet | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb) | 73.68% |
| CIFAR-10 | VGG-16 | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16.ipynb) | 76.31% |
| CIFAR-10 | VGG-19 | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg19.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb) | 74.56% |
| CIFAR-10 | DenseNet-121 | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-densenet121-cifar10.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-densenet121-cifar10.ipynb) | 74.97% |
| CIFAR-10 | ResNet-101 | 图像分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet101-cifar10.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-cifar10.ipynb) | 75.15% |
| MNIST | ResNet 残差模块练习 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/resnet-ex-1.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/resnet-ex-1.ipynb) | 97.91% |
| MNIST | LeNet-5 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-lenet5-mnist.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-mnist.ipynb) | 98.47% |
| MNIST | ResNet-18 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb) | 99.06% |
| MNIST | ResNet-34 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb) | 99.04% |
| MNIST | ResNet-50 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb) | 98.39% |
| MNIST | DenseNet-121 | 数字分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb) | 98.95% |
| CelebA | VGG-16 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb) | 95.48% |
| CelebA | ResNet-18 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb) | 97.38% |
| CelebA | ResNet-34 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb) | 97.56% |
| CelebA | ResNet-50 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb) | 97.40% |
| CelebA | ResNet-101 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb) | 97.52% |
| CelebA | ResNet-152 | 性别分类 | PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb) |  |

## 度量学习

- 多层感知器实现的孪生网络 [TensorFlow 1: [GitHub](tensorflow1_ipynb/metric/siamese-1.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/metric/siamese-1.ipynb)]

## 自编码器

#### 全连接自编码器

- 自编码器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-basic.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-basic.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb)]

#### 卷积自编码器

- 反卷积 / 转置卷积实现的卷积自编码器[TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-deconv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-deconv.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-deconv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb)]
- 转置卷积实现的卷积自编码器（没有使用池化操作） [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb)]
- 最近邻插值实现的卷积自编码器 [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb)]
- 在 CelebA 上训练的最近邻插值卷积自编码器 [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb)]
- 在 Quickdraw 上训练的最近邻插值卷积自编码器 [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb)]

#### 变分自动编码器

- 变分自动编码器 [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-var.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb)]
- 卷积变分自动编码器 [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-var.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-var.ipynb)]

#### 条件变分自动编码器

- 条件变分自动编码器（重建损失中带标签） [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cvae.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae.ipynb)]
- 条件变分自动编码器（重建损失中没有标签） [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb)]
- 卷积条件变分自动编码器（重建损失中带标签） [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb)]
- 卷积条件变分自动编码器（重建损失中没有标签） [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb)]

## 生成对抗网络 (GANs)

- 在 MNIST 上训练的全连接 GAN [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/gan/gan.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/gan/gan.ipynb)]
- 在 MNIST 上训练的全连接 Wasserstein GAN [PyTorch: [GitHub](pytorch_ipynb/gan/wgan-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/gan/wgan-1.ipynb)]
- 在 MNIST 上训练的卷积 GAN [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan-conv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan-conv.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/gan/gan-conv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv.ipynb)]
- 在 MNIST 上使用标签平滑训练的卷积 GAN [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan-conv-smoothing.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan-conv-smoothing.ipynb)] [PyTorch: [GitHub](pytorch_ipynb/gan/gan-conv-smoothing.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv-smoothing.ipynb)]
- 在 MNIST 上训练的卷积 Wasserstein GAN [PyTorch: [GitHub](pytorch_ipynb/gan/dc-wgan-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/gan/dc-wgan-1.ipynb)]

## 图神经网络 (GNNs)

- Most Basic Graph Neural Network with Gaussian Filter on MNIST    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-1.ipynb)]
- Basic Graph Neural Network with Edge Prediction on MNIST    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-edge-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-edge-1.ipynb)]
- Basic Graph Neural Network with Spectral Graph Convolution on MNIST  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-graph-spectral-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-graph-spectral-1.ipynb)]

## 递归神经网络 (RNNs)

#### 多对一：情感分析、分类

- 一个简单的单层RNN（IMDB）[PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_simple_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_imdb.ipynb)]
- 一个简单的单层RNN，带有打包序列，用于忽略填充字符（IMDB） [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb)]
- 带有长短期记忆（LSTM）的RNN（IMDB） [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb)]
- 带有长短期记忆（LSTM）的RNN，使用预训练 GloVe 词向量 [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb)]
- 带有长短期记忆（LSTM）的RNN，训练 CSV 格式的数据集（IMDB）[PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]
- 带有门控单元（GRU）的RNN（IMDB） [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]
- 多层双向RNN（IMDB） [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]

#### 多对多 / 序列对序列

- Char-RNN 实现的文本生成器（Charles Dickens） [PyTorch: [GitHub](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]

## 序数回归

- 序数回归 CNN -- CORAL w. ResNet34（AFAD-Lite） [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb)]
- 序数回归 CNN -- Niu et al. 2016 w. ResNet34（AFAD-Lite） [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb)]
- 序数回归 CNN -- Beckham and Pal 2016 w. ResNet34（AFAD-Lite） [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-beckham2016-afadlite.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-beckham2016-afadlite.ipynb)]

## 技巧和窍门

- 循环学习率 [PyTorch: [GitHub](pytorch_ipynb/tricks/cyclical-learning-rate.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/tricks/cyclical-learning-rate.ipynb)]
- 动态增加 Batch Size 来模拟退火（在 CIFAR-10 上训练 AlexNet） [PyTorch: [GitHub](pytorch_ipynb/tricks/cnn-alexnet-cifar10-batchincrease.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/tricks/cnn-alexnet-cifar10-batchincrease.ipynb)]
- 梯度裁剪（在 MNIST 上训练 MLP） [PyTorch: [GitHub](pytorch_ipynb/tricks/gradclipping_mlp.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/tricks/gradclipping_mlp.ipynb)]

## PyTorch 工作流程和机制

#### 自定义数据集

- 使用 torch.utils.data 加载自定义数据集 -- CSV 文件转换为 HDF5 格式 [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自 CelebA 的面部图像 [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自 Quickdraw 的手绘图像 [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自街景门牌号数据集（SVHN）的图像 [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 亚洲面部数据集 (AFAD) [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-afad.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-afad.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 照片年代追溯数据集（Dating Historical Color Images） [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader_dating-historical-color-images.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader_dating-historical-color-images.ipynb)]

#### 训练和预处理

- 生成训练集和验证集 [PyTorch: [GitHub](pytorch_ipynb/mechanics/validation-splits.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/validation-splits.ipynb)]
- 在 DataLoader 中使用固定内存（pin_memory）技术 [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb)]
- 标准化图像（Standardization） [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-standardized.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-standardized.ipynb)]
- 使用 torchvision 进行图像变换（数据增强） [PyTorch: [GitHub](pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb)]
- 在自己的文本数据上训练 Char-RNN [PyTorch: [GitHub](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]
- 在自己的文本数据集上使用 LSTM 进行情感分类 [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]

#### 并行计算

- 使用 DataParallel 进行多 GPU 训练 -- 在 CelebA 上使用 VGG-16 训练性别分类器 [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb)]

#### 其他

- Sequential API 和 Hook 技术  [PyTorch: [GitHub](pytorch_ipynb/mechanics/mlp-sequential.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/mlp-sequential.ipynb)]
- 同层权值共享  [PyTorch: [GitHub](pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb)]
- 使用 Matplotlib 在 Jupyter Notebook 中绘制实时训练曲线 [PyTorch: [GitHub](pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb)]

#### Autograd

- 在 PyTorch 中获取中间变量的梯度 [PyTorch: [GitHub](pytorch_ipynb/mechanics/manual-gradients.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/pytorch_ipynb/mechanics/manual-gradients.ipynb)]

## TensorFlow 工作流程和机制

#### 自定义数据集

- 使用 NumPy npz 格式打包小批量图像数据集 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb)]
- 使用 HDF5 格式保存小批量图像数据集 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb)]
- 使用输入管道在 TFRecords 文件中读取数据 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/tfrecords.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/tfrecords.ipynb)]
- 使用队列运行器（Queue Runners）从硬盘中直接读取图像 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/file-queues.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/file-queues.ipynb)]
- 使用 TensorFlow 数据集 API [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/dataset-api.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/dataset-api.ipynb)]

#### 训练和预处理

- 保存和加载模型 -- 保存为 TensorFlow Checkpoint 文件和 NumPy npz 文件 [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb) \| [Nbviewer](https://nbviewer.jupyter.org/github/ypwhs/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb)]
