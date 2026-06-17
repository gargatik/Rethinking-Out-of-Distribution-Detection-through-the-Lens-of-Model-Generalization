# Rethinking Out-of-Distribution Detection through the Lens of Model Generalization

This repository contains the official implementation for the paper **"Rethinking Out-of-Distribution Detection through the Lens of Model Generalization"**, accepted at **ICME 2026**.

📄 **[Read the Paper](https://people.cs.nycu.edu.tw/~yushuen/data/RethinkingOOD26.pdf)** ---

## 📋 Overview
Our work explores how Self-Supervised Learning (SSL) representations and pseudo-labeling strategies affect a model's ability to generalize to In-Distribution (ID) data while improving the features required for Out-of-Distribution (OOD) detection. We leverage a mixed Barlow Twins framework coupled with clustering-driven pseudo-labeling to enhance downstream OOD detection performance.

---

## 🛠️ Environment & Setup
The environment requirements are identical to the [Mixed Barlow Twins repository](https://github.com/wgcban/mix-bt). Ensure you have the proper dependencies installed for PyTorch, torchvision, and scikit-learn.

---

## 🚀 Pipeline Workflow

### 1. Self-Supervised Pre-Training (SSL)
We employ the Mixed Barlow Twins method for representation learning. To start pre-training and obtain k-NN evaluation results on **CIFAR-10, CIFAR-100, TinyImageNet, and STL-10** using **ResNet-18** or **ResNet-50** backbones, navigate to the `SSL` folder and execute the script matching your target configuration:

```bash
# Pre-training with ResNet-18
sh scripts-pretrain-resnet18/[dataset].sh
```bash
# Pre-training with ResNet-50
sh scripts-pretrain-resnet50/[dataset].sh

### Pseudo-Labeling, Training, & Evaluation Workflow

#### 2. Pseudo-Label Generation
To generate pseudo-labels using K-Means clustering, run the `clustering.py` script located inside the `SSL` folder:
```bash
python clustering.py

#### 2. Pseudo-Label Training
To train your backbone using the generated pseudo-labels, configure your data paths and execute the training script:

Open Pseudo_label_training/utils/data.py.

Replace the placeholder paths for the train and test pseudo-labels with your actual generated file paths.

```bash

python train_backbone.py -d cifar10 -g 0 -n resnet18 -s 'save_name'

#### 2. Evaluation & OOD Detection

Single-Model OOD Detection
To run Out-of-Distribution detection using the norm of the penultimate block (Feature Norm method), pass the -m featurenorm flag to eval.py:
Pre-trained Checkpoints
Pre-trained backbones and evaluation checkpoints will be made available for download here:

```bash
python eval.py -n resnet18 -d 'data_name' -g 'gpu_num' -s 'save_name' -m featurenorm

Ensemble OOD Detection
To use an ensemble configuration for OOD detection, execute the following script:

python ensemblel.py -n resnet18 -d cifar10 -g 0 -s 'save_name'

Acknowledgments
This implementation builds upon or references core components from the following open-source repositories:

Self-Supervised Learning Framework: https://github.com/wgcban/mix-bt

OOD Evaluation & Block Architectures: https://github.com/gist-ailab/block-selection-for-OOD-detection