# Breast Cancer Detection Using Deep Learning

## Overview

This project explores the use of deep learning and computer vision to detect breast cancer from mammogram images. Because medical scans often contain extraneous artifacts—such as clinical text labels, orientation markers, and scanner noise—this pipeline combines automated anatomical preprocessing with a diverse suite of Convolutional Neural Networks (CNNs) and ensemble learning to maximize diagnostic reliability.

## Key Methodology

### 1. Reproducible Experimental Setup

To ensure consistent and scientifically valid results, the environment enforces deterministic behavior across all random number generators (`SEED = 42`), guaranteeing identical data splits and model initialization across runs.

### 2. Artifact & Background Removal (`RemoveBackgroundNoCrop`)

Raw mammograms frequently include bright metadata tags in the background that can bias a deep learning model. To solve this without distorting the anatomical proportions of the scan, this project implements a custom computer vision preprocessing step:

* **Otsu's Thresholding:** Automatically separates the foreground breast tissue from the dark scan background.
* **Contour Isolation:** Identifies the largest continuous contour (the breast itself) and masks out smaller, disconnected regions like text labels and scanner noise.
* **Spatial Preservation:** Applies the mask in-place without cropping, maintaining the original aspect ratio and spatial context of the tissue before feeding it into standard 3-channel neural network architectures.

### 3. Visual Verification

Before model training occurs, the pipeline generates side-by-side diagnostic comparisons of the raw and preprocessed scans to verify that artifacts are successfully stripped away while preserving critical tissue boundaries.

## Model Architectures & Ensembling

To capture a wide spectrum of visual features—from fine microcalcifications to broader tissue density patterns—five distinct CNN architectures were evaluated:

* **ResNet-50:** Utilizes residual skip connections to train deep representations without vanishing gradients.
* **DenseNet-121:** Connects each layer to every other layer in a feed-forward fashion, encouraging feature reuse—particularly effective for medical imaging.
* **VGG-16:** A classic deep architecture using small $3 \times 3$ convolutional filters to extract hierarchical spatial features.
* **ConvNeXt (`convnext_small`):** A modern pure-ConvNet architecture designed with Vision Transformer principles for strong global and local feature extraction.
* **EfficientNet (`efficientnet_v2_s`):** Uses compound scaling and progressive learning to achieve high accuracy and parameter efficiency.

### Ensemble Strategy (Soft & Hard Voting)

Rather than relying on a single network, the top-performing models (retrained on combined training and validation data with tuned dropout regularization) are grouped into two specialized ensembles:

1. **Best Accuracy Ensemble:** Combines `convnext_small` (dropout `0.4`), `densenet121` (dropout `0.4`), and `efficientnet_v2_s` (dropout `0.2`) to maximize overall classification accuracy and ROC-AUC.
2. **Best F1 Ensemble:** Combines `densenet121` (dropout `0.2`), `efficientnet_v2_s` (dropout `0.2`), and `vgg16` (dropout `0.2`) to optimize the balance between precision and recall—critical in medical diagnostics to minimize false negatives.

Both ensembles are evaluated using two decision-aggregation techniques:

* **Soft Voting:** Averages the predicted class probabilities across all models in the ensemble before making the final classification.
* **Hard Voting:** Uses majority-rule consensus based on each individual model's discrete class prediction.

## Experimental Findings & Evaluation Results

### Hyperparameter Tuning Insights

Across hyperparameter tuning, a clear trade-off emerged between overall accuracy and minority-class sensitivity:
* **F1-oriented configurations** typically leverage class weighting, lifting minority-class recall and F1-score.
* **Accuracy-oriented configurations** generally omit class weighting, raising overall accuracy at the expense of lower F1-scores.

### Single-Model Performance Analysis

* **EfficientNet (`efficientnet_v2_s`):** The strongest standalone backbone. Its *Best Accuracy* variant achieved the highest single-model accuracy (**76.19%**) and AUC (**0.8078**), making it a strong high-specificity option. Its *Best F1* variant prioritized sensitivity (**recall: 0.6797**) and delivered the highest single-model F1-score (**0.6667**).
* **ConvNeXt (`convnext_small`):** The second-strongest performer and a reliable all-rounder. Its *Best Accuracy* setup achieved **73.81%** accuracy and **0.7703** AUC—and notably outperformed its *Best F1* variant in recall and F1 (**0.6239**), benefiting strongly from its regularization and layer-freezing configuration.
* **DenseNet-121:** Maintained consistent accuracy (**69.35%**) across both configurations while showing a clear sensitivity boost under *Best F1* (**recall: 0.5703** vs. **0.4297**; **F1: 0.5863** vs. **0.5164**; **AUC: 0.7520**).
* **ResNet-50:** Served as a predictable baseline—*Best Accuracy* favored precision (**0.6277**) and accuracy (**69.05%**), while *Best F1* lifted recall (**0.5469**), F1 (**0.5600**), and AUC (**0.7462** vs. **0.7367**).
* **VGG-16:** Performed best under the *Best F1* configuration (**69.64%** accuracy, **0.5469** recall, **0.5785** F1, **0.7389** AUC), whereas its *Best Accuracy* variant leaned too heavily toward the majority class.

### Ensemble Evaluation

#### Best Accuracy Group

| Model / Strategy | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| `convnext_small` | 73.81% | 0.6239 | 0.7703 |
| `densenet121` | 69.35% | 0.5164 | 0.7483 |
| `efficientnet_v2_s` | **76.19%** | **0.6296** | 0.8078 |
| **Ensemble (Soft Voting)** | 75.89% | 0.6087 | **0.8220** |
| **Ensemble (Hard Voting)** | 75.60% | 0.6019 | **0.8220** |

#### Best F1 Group

| Model / Strategy | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| `densenet121` | 69.35% | 0.5863 | 0.7520 |
| `efficientnet_v2_s` | 74.11% | 0.6667 | **0.8027** |
| `vgg16` | 69.64% | 0.5785 | 0.7389 |
| **Ensemble (Soft Voting)** | **76.19%** | **0.6800** | 0.7973 |
| **Ensemble (Hard Voting)** | 73.81% | 0.6508 | 0.7973 |

### Key Takeaways

* **Soft Voting Outperforms Hard Voting:** Averaging predicted probabilities consistently yielded higher F1-scores than discrete majority voting while matching or exceeding accuracy.
* **Best Accuracy (Soft Vote):** Achieved the highest overall ROC-AUC (**0.8220**) alongside **79.75%** precision and **75.89%** accuracy—ideal when specificity and ranking discrimination are prioritized.
* **Best F1 (Soft Vote):** Selected as the best overall deployment candidate, combining peak accuracy (**76.19%**) with the highest F1-score (**0.6800**) and strong sensitivity (**recall: 0.6641**) to minimize missed cancer cases.

## Future Improvements

* **Multi-Seed Retraining:** Train the final ensemble configurations across three distinct random seeds (with fixed data splits and hyperparameters) to evaluate variance and retain the highest-performing checkpoint.
* **Extended Training Schedules:** Increase the maximum epoch budget while keeping early stopping on validation F1, `ReduceLROnPlateau`, and best-epoch checkpointing—adopting the longer schedule only if it improves both F1 and accuracy.
* **Systematic Block Freezing:** Explore structured, multi-stage layer-freezing strategies across backbones rather than a single freeze configuration to further improve feature generalization.

## Technologies Used

* **Deep Learning Framework:** PyTorch & Torchvision
* **Computer Vision & Image Processing:** OpenCV, Pillow (PIL)
* **Data Analysis & Visualization:** NumPy, Pandas, Scikit-Learn, Matplotlib