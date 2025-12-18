# Rock Classification

## Description
This project is a Final Examination (UAS) assignment for the Deep Learning course, focusing on rock image classification using deep learning techniques. The system is developed using Convolutional Neural Networks (CNN) and a ResNet34 architecture implemented with PyTorch, and is integrated into an application interface built using Streamlit.

The application accepts rock image inputs, performs preprocessing steps, and generates rock class predictions based on trained deep learning models.

---

## Objectives
The objectives of this project are:
1. To implement deep learning algorithms for image classification.
2. To compare the performance of a Custom CNN and a ResNet34 model.
3. To apply the concepts of model training, evaluation, and deployment.
4. To demonstrate the application of deep learning methods in geological image classification.

---

## Models Used
1. **Custom Convolutional Neural Network (CNN)**  
   A CNN model designed and trained from scratch to suit the characteristics of rock image data.

2. **ResNet34 (Transfer Learning)**  
   A ResNet34 model pretrained on ImageNet, utilized to improve generalization capability and classification accuracy.

The trained models are stored in `.pth` format and managed using Git Large File Storage (Git LFS).

---

## Directory Structure
```
rock-classification/
│
├── TRAIN_CUSTOM_CNN/          # Custom CNN training scripts
├── TRAIN_RESNET/              # ResNet34 training scripts
│
├── best_model_custom.pth      # Custom CNN model (Git LFS)
├── best_model_resnet34.pth    # ResNet34 model (Git LFS)
│
├── class_names.txt            # Rock class labels
├── host1.py                   # Streamlit application
│
├── .gitignore
├── .gitattributes             # Git LFS configuration
└── README.md
```

---

## How to Run the Application

1. Clone the repository:
```bash
git clone https://github.com/ipul122/rock-classification.git
cd rock-classification
```

2. Download Model:
```bash
https://bit.ly/48MH0yE
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run host1.py
```

---

## Tools and Libraries
- Python 3.x
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy

---

## Academic Context
This project was developed as part of the assessment for the Deep Learning course and aims to demonstrate the student’s understanding of deep learning implementation, particularly convolutional neural networks and transfer learning techniques, applied to rock image classification.

---
