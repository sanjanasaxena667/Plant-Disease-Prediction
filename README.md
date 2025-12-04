README.

🌱 PlantVillage Leaf Disease Classification
Deep Learning Model for Detecting Plant Leaf Diseases Using CNN
📌 Project Overview

This repository contains a single Jupyter Notebook, plantvillage-classification.ipynb, which implements an end-to-end deep learning pipeline for identifying plant leaf diseases using the popular PlantVillage dataset.

The goal of this project is to create an automated, high-accuracy disease classification model using Convolutional Neural Networks (CNNs), enabling early diagnosis and improving agricultural productivity.

📁 Repository Structure
.
├── plantvillage-classification.ipynb   # Main notebook (training + evaluation)
└── README.md                           # Project documentation


⚠ Note: The dataset and model files are not included in this repository due to their large size. Instructions to set them up are provided below.

⭐ Key Features

✔ Complete image classification pipeline in a single notebook
✔ CNN / Transfer Learning-based architecture (depending on notebook choice)
✔ GPU-accelerated training support
✔ Data preprocessing + augmentation built-in
✔ Evaluation metrics, accuracy curves, and confusion matrix
✔ Ready for deployment or further model experimentation

📦 Setup Instructions
1️⃣ Create a Python Virtual Environment
python -m venv venv

2️⃣ Activate the Environment

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3️⃣ Install Required Libraries

Inside the activated environment:

pip install tensorflow keras numpy matplotlib seaborn scikit-learn pillow


If your notebook uses additional libraries, install them as needed.

🗂 Dataset Setup (PlantVillage)

Download dataset from the official source:
https://www.kaggle.com/datasets/emmarex/plantdisease

Extract it into a folder named:

data/


Ensure the images are structured like:

data/
 ├── train/
 ├── test/
 └── validation/


If the notebook performs splitting automatically, you only need:

data/
 └── PlantVillage/

🧠 Model Workflow (Inside Notebook)

The notebook consists of:

1️⃣ Data Loading & Preprocessing

Reading images

Normalizing pixel values

Applying augmentation

Creating train/validation/test generators

2️⃣ Model Architecture

Uses either:

Custom CNN
or

Transfer learning model such as MobileNetV2 / EfficientNetB0

Includes:

Convolution layers

MaxPooling

Dropout

Fully connected classification head

3️⃣ Training

Adam optimizer

Categorical cross-entropy

EarlyStopping & ModelCheckpoint callbacks (if used)

Epoch-by-epoch visualization

4️⃣ Evaluation

Accuracy & loss curves

Confusion matrix

Classification report

5️⃣ Prediction

Sample code included for testing the model on new leaf images.

📊 Expected Output

The notebook produces:

Training accuracy & validation accuracy graphs

Final test accuracy (typically 95%–99% depending on model/dataset quality)

Confusion matrix for all classes

Sample prediction results

🛠 How to Run the Notebook

Run the following after creating the environment and installing dependencies:

jupyter notebook


Then open:

plantvillage-classification.ipynb

🚀 Future Enhancements

🔹 Deploy as a web or mobile app
🔹 Convert model to TensorFlow Lite / ONNX
🔹 Integrate Grad-CAM for visualization
🔹 Improve dataset with real-world images