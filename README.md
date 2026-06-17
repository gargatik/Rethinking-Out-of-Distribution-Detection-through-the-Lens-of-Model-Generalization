# Rethinking Out-of-Distribution Detection through the Lens of Model Generalization

Official implementation of the paper:

**Rethinking Out-of-Distribution Detection through the Lens of Model Generalization**
Accepted at **ICME 2026**

📄 **Paper:** https://people.cs.nycu.edu.tw/~yushuen/data/RethinkingOOD26.pdf

---

## Overview

Out-of-Distribution (OOD) detection aims to identify samples that do not belong to the training distribution. While most existing methods focus on designing specialized OOD scoring functions, we revisit the problem from the perspective of **representation learning and model generalization**.

Our framework investigates how **Self-Supervised Learning (SSL)** representations and **pseudo-labeling strategies** affect downstream OOD detection performance. Specifically, we leverage a **Mixed Barlow Twins** framework together with **clustering-based pseudo-label generation** to learn more discriminative representations that improve both in-distribution generalization and OOD detection.

---

## Repository Structure

```text
.
├── SSL/
│   ├── clustering.py
│   ├── scripts-pretrain-resnet18/
│   └── scripts-pretrain-resnet50/
│
├── Pseudo_label_training/
│   ├── train_backbone.py
│   └── utils/
│       └── dataset.py
│
├── eval.py
├── ensemble.py
└── README.md
```

---

## Environment Setup

The environment requirements are largely identical to the Mixed Barlow Twins implementation.

### Install Dependencies

```bash
pip install torch torchvision scikit-learn numpy tqdm
```

Additional package requirements can be found in the original Mixed Barlow Twins repository:

https://github.com/wgcban/mix-bt

---

## Pipeline Workflow

### Step 1: Self-Supervised Pre-Training

We first perform self-supervised representation learning using the Mixed Barlow Twins framework.

Navigate to the `SSL` directory and launch the corresponding training script.

#### ResNet-18

```bash
sh scripts-pretrain-resnet18/[dataset].sh
```

#### ResNet-50

```bash
sh scripts-pretrain-resnet50/[dataset].sh
```

Supported datasets include:

* CIFAR-10
* CIFAR-100
* SVHN
* etc.

The scripts will train the SSL model and report k-NN evaluation performance.

---

### Step 2: Generate Pseudo Labels

After SSL pre-training, generate pseudo labels using K-Means clustering.

```bash
cd SSL

python clustering.py
```

This script extracts features from the pretrained SSL backbone and generates pseudo labels for downstream training.

---

### Step 3: Pseudo-Label Training

Before training, configure the pseudo-label paths in:

```text
Pseudo_label_training/utils/dataset.py
```

Replace the placeholder paths with the generated training and testing pseudo-label files.

Then train the backbone:

```bash
python train_backbone.py \
    -d cifar10 \
    -g 0 \
    -n resnet18 \
    -s save_name
```

#### Arguments

| Argument | Description          |
| -------- | -------------------- |
| `-d`     | Dataset name         |
| `-g`     | GPU ID               |
| `-n`     | Network architecture |
| `-s`     | Experiment/save name |

---

### Step 4: OOD Evaluation

#### Feature Norm OOD Detection

To perform OOD detection using the feature norm of the penultimate layer:

```bash
python eval.py \
    -n resnet18 \
    -d cifar10 \
    -g 0 \
    -s save_name \
    -m featurenorm
```

---

### Ensemble OOD Detection

To evaluate using the ensemble strategy:

```bash
python ensemble.py \
    -n resnet18 \
    -d cifar10 \
    -g 0 \
    -s save_name
```

---

## Pretrained Models

Pretrained checkpoints and evaluation models will be released soon.

```text
Coming Soon
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{garg2026rethinking,
  title={Rethinking Out-of-Distribution Detection through the Lens of Model Generalization},
  author={Garg, Atik and others},
  booktitle={Proceedings of the IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}
```

---

## Acknowledgments

This work builds upon the following open-source projects:

### Mixed Barlow Twins

Repository:
https://github.com/wgcban/mix-bt

### Block Selection for OOD Detection

Repository:
https://github.com/gist-ailab/block-selection-for-OOD-detection

We sincerely thank the authors for making their code publicly available.

---

## Contact

For questions, issues, or collaborations, please open an issue in this repository.
