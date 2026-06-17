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

# Pre-training with ResNet-50
sh scripts-pretrain-resnet50/[dataset].sh