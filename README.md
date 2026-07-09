# Non-Intrusive Eye Emotion Detection (Hybrid CNN + ViT)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)

## 📌 Overview
This project implements a robust, non-intrusive deep learning system designed to classify human emotions purely from images of eyes. Utilizing a sophisticated hybrid architecture that combines standard Convolutional Neural Networks (CNNs), a pre-trained ResNet18, and a Vision Transformer (ViT), this capstone-grade model aims for high accuracy in nuanced facial analysis. 

Additionally, the project features **Explainable AI (XAI)** via Grad-CAM, allowing users to visually interpret which regions of the eye the model focuses on to make its predictions.

## 📊 Dataset
The model is trained on the [Eye Emotion Dataset](https://www.kaggle.com/datasets/mdnymurrahmanshuvo/eye-emotion-dataset-diu), which is automatically downloaded via `kagglehub` during execution.

It detects the following **6 emotion classes**:
* `Angry`
* `Disgust`
* `Fear`
* `Happy`
* `Sad`
* `Surprise`

## 🧠 Hybrid Model Architecture
To capture both local textures and global context within the eye images, the model uses a three-branch fusion strategy:

1. **Custom CNN Branch:** A 3-block convolutional network (with BatchNorm and MaxPool) designed to extract low-level, dataset-specific edge and texture features.
2. **ResNet18 Branch:** A pre-trained residual network utilized for robust, mid-level feature extraction.
3. **Vision Transformer (ViT) Branch:** Uses `vit_small_patch16_224` (via `timm`) to capture global spatial dependencies and long-range visual context.

**Fusion Layer:** The feature maps from all three branches are concatenated (1024 dimensions) and passed through a dense classifier with dropout (0.4) to prevent overfitting.

## 🔍 Explainable AI (Grad-CAM)
Trust and transparency are critical in emotion detection. This project includes a custom `GradCAM` implementation hooked to the final convolutional layer of the ResNet18 branch. 

When testing an image, the model outputs:
1. The original eye image.
2. A Grad-CAM attention heatmap.
3. A superimposed explainable result showing exactly where the model "looked" to determine the emotion (highlighted in red).

## ⚙️ Requirements & Installation
   ```bash
   git clone [https://github.com/Siva-6030/Non-Intrusive-Emotion-Detection.git](https://github.com/Siva-6030/Non-Intrusive-Emotion-Detection.git)
   cd Non-Intrusive-Emotion-Detection
