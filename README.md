# XAI
This repository contains the code and resources used in the "Explainable AI" project. 
The goal of this project is to apply five XAI techniques to understand the decision-making process of Convolutional Neural Networks (CNNs) in classifying chest X-ray images with "Pneumonia" or "No Pneumonia".

**Data Preprocessing:** The MedMNIST dataset is preprocessed, including normalization, data augmentation, and splitting into training, validation, and test sets.

**Model Training:** Two pre-trained CNN architectures, VGG16 and ResNet18, are fine-tuned on the preprocessed MedMNIST dataset for the pneumonia classification task.

**XAI Techniques:** Five different XAI methods are implemented and applied to the trained models to provide interpretable insights into their decision-making process:
* Grad-CAM (Gradient-weighted Class Activation Mapping)
* Integrated Gradients
* SHAP (SHapley Additive exPlanations)
* Occlusion Sensitivity
* RISE (Random Input Sampling for Explanation)

**Visualization and Analysis:** The XAI outputs, such as heatmaps and saliency maps, are visualized and analyzed to understand how the models focus on relevant regions in the chest X-ray images for making their predictions.

<br>
_Code Structure_

**Our Implementation** Preliminary_GradCAM.ipynb, VGG_16_RISE.py, RESNET_18_RISE.py

**VGG16 training** VGG_16.ipynb

**VGG16 with Builtin Explainable AI models** VGG_16_IntegratedGradients.py, VGG_16_grad_cam.py (and VGG_16_gradcam_overlay), VGG_16_occlusionsensitive.py, VGG_16_shap.py

**ResNet18 with Builtin Explainable AI models** ResNet18_XAIs.ipynb



<br>
Folders have example image results used in the poster and final report.
