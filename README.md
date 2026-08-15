# 🫁 Chest X-ray Pneumonia Detection | Deep Learning Project

<p align="center">

**CNN-Based Medical Image Classification**

A deep learning application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a Convolutional Neural Network (CNN).

[🚀 Live Demo](https://chest-xray-pneumonia-detection-03.streamlit.app/)

</p>

---

## 📌 Project Overview

Pneumonia is a serious respiratory infection that can appear as abnormalities on chest X-ray images.

This project demonstrates how **Deep Learning, Computer Vision, and Healthcare AI** can be combined to develop an image classification system capable of distinguishing between **NORMAL** and **PNEUMONIA** chest X-ray images.

The project covers the complete machine-learning workflow:

```text
Medical Images
      ↓
Data Preparation
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
Class Imbalance Handling
      ↓
CNN Model Development
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Streamlit Application
      ↓
Cloud Deployment

The trained model is integrated into an interactive Streamlit web application,
 allowing users to upload a chest X-ray image and obtain a model prediction.

⚠️ Medical Disclaimer: This project is intended strictly for educational,
 research, and portfolio demonstration purposes. It is not a clinically validated
diagnostic system and must not be used to diagnose or make medical decisions about patients.

🚀 Live Demo
🫁 Try the Application

Launch Chest X-ray Pneumonia Detection App →

The application allows users to:

📤 Upload a chest X-ray image
🖼️ Preview the uploaded image
🧠 Run the trained CNN model
🔍 Receive a predicted class
📊 View the estimated pneumonia probability
🎯 Project Objectives

The main objectives of this project were to:

* Build a CNN for chest X-ray image classification.
* Develop an image preprocessing pipeline.
* Normalize image pixel values.
* Resize images for CNN input.
* Address class imbalance using class weights.
* Train and evaluate a deep learning model.
* Analyze classification performance using multiple metrics.
* Evaluate model discrimination using ROC-AUC.
* Build an interactive Streamlit application.
* Deploy the trained model as a cloud-based application.
* Demonstrate an end-to-end Healthcare AI workflow.


📂 Dataset

# The project uses a Chest X-ray Pneumonia dataset containing two image classes:

NORMAL
PNEUMONIA


The training dataset contains substantially more Pneumonia images than Normal images.

To address this class imbalance, class weights were incorporated during model training.


Class Weights

| Class                                               | Weight |
| ---------                                           | -----: |
| NORMAL                                              | 1.9448 |
| PNEUMONIA                                           | 0.6730 |


🖼️ Image Configuration


| Parameter                          |     Value |
| -------------------                | --------: |
| Image Size                         | 224 × 224 |
| Image Channels                     |         3 |
| Batch Size                         |        32 |
| Random Seed                        |        42 |
| Classification Type                |    Binary |
| Pixel Value Range                  | 0.0 – 1.0 |



Images were resized to 224 × 224 pixels and normalized so that pixel values were scaled to the range 0–1.


🧠 Deep Learning Model


A Convolutional Neural Network (CNN) was developed using TensorFlow/Keras for binary image classification.

| Specification                                       |                        Value |
| ------------------------                            | ---------------------------: |
| Model Type                                          | Convolutional Neural Network |
| Input Shape                                         |                224 × 224 × 3 |
| Total Parameters                                    |                   11,169,089 |
| Trainable Parameters                                |                   11,169,089 |
| Non-Trainable Parameters                            |                            0 |
| Model Format                                        |                     `.keras` |
| Model File                                          |   `best_pneumonia_cnn.keras` |


The trained CNN model was approximately 134 MB and was managed using Git LFS in the GitHub repository.



⚙️ Image Preprocessing

The image preprocessing pipeline included:

1. Loading chest X-ray images.
2. Resizing images to 224 × 224.
3. Converting images to RGB.
4. Normalizing pixel values to the 0–1 range.
5. Creating training and validation datasets.
6. Applying class weights to address class imbalance.
7. Applying image augmentation during model training.
8. Preprocessing Pipeline
9. Chest X-ray Image
        ↓
Image Loading
        ↓
Resize to 224 × 224
        ↓
RGB Conversion
        ↓
Pixel Normalization
        ↓
Data Augmentation
        ↓
CNN Input



🏋️ Model Training

The CNN model was trained using TensorFlow/Keras.

| Parameter                                      | Configuration      |
| -------------------                            | ------------------ |
| Framework                                      | TensorFlow / Keras |
| Model                                          | CNN                |
| Image Size                                     | 224 × 224          |
| Batch Size                                     | 32                 |
| Maximum Epochs                                 | 20                 |
| Optimizer                                      | Adam               |
| Classification                                 | Binary             |
| Class Imbalance                                | Class Weights      |
| Model Checkpointing                            | Enabled            |


The best-performing model checkpoint was saved as:

best_pneumonia_cnn.keras



📊 Model Evaluation

The trained model was evaluated on a separate test dataset containing 624 chest X-ray images.

The evaluation included:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC


🏆 Model Performance

The model achieved the following performance on the test dataset:

| Metric                                           |     Result |
| -----------------------                          | ---------: |
| **Test Accuracy**                                | **81.57%** |
| **Test ROC-AUC**                                 | **95.66%** |
| **Pneumonia Precision**                          | **77.78%** |
| **Pneumonia Recall**                             | **98.72%** |
| **Pneumonia F1-score**                           | **87.01%** |
| **Normal Precision**                             | **96.12%** |
| **Normal Recall**                                | **52.99%** |
| **Normal F1-score**                              | **68.32%** |




📋 Classification Report

              precision       recall     f1-score      support


NORMAL          0.9612         0.5299      0.6832         234
PNEUMONIA       0.7778         0.9872      0.8701         390


accuracy                                    0.8157         624


macro avg       0.8695          0.7585       0.7766        624
weighted avg    0.8466          0.8157       0.8000        624



🧩 Confusion Matrix

The model produced the following confusion matrix on the test set:

                                Predicted
              NORMAL  PNEUMONIA


| Actual Class |               Predicted NORMAL |         Predicted PNEUMONIA |
| ------------ |               ---------------: |         ------------------: |
| NORMAL       |                            124 |                         110 |
| PNEUMONIA    |                              5 |                         385 |




The model correctly classified:

124 NORMAL images.
385 PNEUMONIA images.



The model incorrectly classified:

110 NORMAL images as PNEUMONIA.
5 PNEUMONIA images as NORMAL.
🔍 Key Model Insights
Strong Pneumonia Recall

The model achieved a 98.72% recall for Pneumonia.

Out of 390 Pneumonia images:



385 → Correctly classified as PNEUMONIA
5   → Classified as NORMAL



This indicates that the model identified the large majority of Pneumonia images in the test dataset.

Strong ROC-AUC

The model achieved a 95.66% test ROC-AUC, demonstrating strong ability to distinguish between the two classes across different classification thresholds.

Normal-Class Limitation

The model achieved a 52.99% recall for NORMAL images.

A total of 110 NORMAL images were classified as PNEUMONIA.

This resulted in a relatively high false-positive rate for the Normal class and is an important limitation of the current model.



🧪 Prediction Testing

The trained model was tested using chest X-ray images from the test dataset.

Example 1 — Pneumonia

Actual Class: PNEUMONIA
Predicted Class: PNEUMONIA
Pneumonia Probability: 99.92%

Example 2 — Normal

Actual Class: NORMAL
Predicted Class: NORMAL
Pneumonia Probability: 38.59%

These tests demonstrate that the deployed prediction pipeline can process both NORMAL and PNEUMONIA chest X-ray images.


🖥️ Streamlit Application

The trained CNN was integrated into an interactive Streamlit web application.

Application Workflow
User Uploads X-ray
        ↓
Image Preview
        ↓
Image Resizing
        ↓
Pixel Normalization
        ↓
CNN Prediction
        ↓
Predicted Class
        ↓
Pneumonia Probability

The application provides a simple interface for demonstrating the trained deep learning model.



🛠️ Technology Stack
Programming
Python 3.12
Deep Learning
TensorFlow
Keras
Convolutional Neural Networks
Image Processing
NumPy
Pillow
Model Evaluation
Scikit-learn
Matplotlib
Application Development
Streamlit
Version Control
Git
GitHub
Git LFS
Development Environment
Google Colab
Jupyter Notebook
Visual Studio Code
Deployment
Streamlit Community Cloud
📦 Requirements

The application uses the following core dependencies:

streamlit==1.48.1
tensorflow==2.21.0
numpy==1.26.4
Pillow==11.3.0

The deployment environment uses:

Python 3.12
📁 Project Structure
Chest-Xray-Pneumonia-Detection/
│
├── app.py
├── best_pneumonia_cnn.keras
├── requirements.txt
├── README.md
├── .gitignore
├── .gitattributes
│
├── images/
│   └── project screenshots
│
├── notebooks/
│   └── CNN training notebook
│
├── models/
│   └── model-related files
│
└── dataset/
    └── local dataset files

The raw dataset and large archive files are excluded from normal Git tracking. The trained CNN model is managed using Git LFS.

💻 Run the Project Locally
1. Clone the Repository
git clone https://github.com/anitaokechukwu/Chest-Xray-Pneumonia-Detection.git
2. Navigate to the Project
cd Chest-Xray-Pneumonia-Detection
3. Create a Virtual Environment
python -m venv venv
4. Activate the Environment
Windows
venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
6. Run the Streamlit Application
python -m streamlit run app.py

The application will open in your default web browser.

☁️ Cloud Deployment

The application was deployed using Streamlit Community Cloud.

🚀 Live Application

🫁 Launch Chest X-ray Pneumonia Detection App →

The trained CNN model is stored in the GitHub repository using Git Large File Storage (Git LFS) because of its large file size.

🔐 Git & Large File Management

The project uses .gitignore to prevent unnecessary and very large files from being committed to the repository.

Examples include:

dataset/
archive*.zip
models/

The trained model:

best_pneumonia_cnn.keras

is managed using:

Git LFS

This allows the large model artifact to remain version-controlled while keeping the regular Git repository lightweight.

⚠️ Project Limitations

This project has several important limitations.

Dataset Generalization

The model was trained and evaluated using a specific dataset. Performance may differ when applied to:

Images from different hospitals
Different X-ray machines
Different imaging protocols
Different patient populations
External datasets
False Positives

The model produced 110 false-positive Pneumonia predictions among Normal X-rays in the test set.

This contributed to the relatively low Normal recall of 52.99%.

Clinical Validation

The model has not undergone clinical validation or prospective clinical testing.

Model Interpretability

The current application does not include explainability techniques such as:

Grad-CAM
Saliency Maps
Integrated Gradients

Therefore, the model's decision-making process is not directly visualized.

🔮 Future Improvements

Potential future improvements include:

Implementing Grad-CAM for visual model explanations.
Experimenting with transfer learning.
Testing architectures such as ResNet, DenseNet, and EfficientNet.
Performing systematic hyperparameter optimization.
Improving Normal-class recall.
Optimizing the classification threshold.
Evaluating the model on external datasets.
Performing cross-validation.
Calibrating prediction probabilities.
Conducting more detailed error analysis.
Evaluating model fairness and generalization.
Exploring model compression for more efficient deployment.
💡 Skills Demonstrated

This project demonstrates practical experience in:

🧠 Deep Learning
👁️ Computer Vision
🏥 Healthcare AI
🩻 Medical Image Classification
🐍 Python
🤖 TensorFlow / Keras
🧮 NumPy
🖼️ Image Preprocessing
⚖️ Class Imbalance Handling
📊 Model Evaluation
📈 ROC-AUC Analysis
🎯 Precision, Recall & F1-score
🧩 Confusion Matrix Analysis
🌐 Streamlit
☁️ Cloud Deployment
🔗 Git & GitHub
📦 Git LFS
📈 End-to-End Project Pipeline
                  ┌─────────────────────┐
                  │   Chest X-ray Data  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Data Preprocessing  │
                  │ 224 × 224 Images    │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Data Augmentation   │
                  │ + Class Weights     │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │      CNN Model      │
                  │   TensorFlow/Keras  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Model Evaluation    │
                  │ Accuracy / Recall   │
                  │ F1 / ROC-AUC        │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │  Trained Model      │
                  │ .keras + Git LFS    │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Streamlit Web App   │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │   Cloud Deployment  │
                  └─────────────────────┘
🏥 Healthcare AI Perspective

This project demonstrates how machine learning can be explored within healthcare workflows for medical image classification.

Rather than evaluating the model using accuracy alone, this project considers:

* Precision
* Recall
* F1-score
* ROC-AUC
* False positives
* False negatives
* Class imbalance
* Model limitations

This provides a broader perspective on evaluating machine learning systems in healthcare-related applications.

👩‍💻 Author
Anita Okechukwu

Healthcare Data Analyst | Data Science | Machine Learning | Healthcare AI

Interested in applying Data Analytics, Machine Learning, Deep Learning, and Cloud Technologies to healthcare problems.


⭐ Project Highlight


Developed and deployed a CNN-based chest X-ray classification system achieving 95.66% test ROC-AUC and 98.72% Pneumonia recall, with an interactive Streamlit application for chest X-ray image classification.


⚠️ Medical Disclaimer


This project is intended solely for educational, research, and portfolio demonstration purposes.

The predictions generated by this application are outputs of a machine-learning model and must not be interpreted as medical diagnoses.

This application is not a substitute for professional medical evaluation, clinical judgment, or diagnostic testing.

Always consult a qualified healthcare professional for medical evaluation and diagnosis.


🔗 Project Links
🚀 Live Streamlit Application
💻 GitHub Repository


### Important before you paste it


I corrected the formatting issue in your current README where the first workflow code block was **not closed with**:


```text


The complete version above has every code block properly opened and closed, so GitHub will render the sections correctly.


After you paste and save it locally, run:


```powershell
git add README.md
git commit -m "Update professional project README"
git push

Then your GitHub repository will have a much more polished, consistent presentation.
