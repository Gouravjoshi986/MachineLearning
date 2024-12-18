# Multiple Disease Prediction Model
This repository contains a machine learning application that predicts three diseases: Diabetes, Heart Disease, and Parkinson's Disease. The model uses Support Vector Machine (SVM) and Logistic Regression for classification, and the deployment is done using Streamlit.

# Table of Contents
Introduction
Features
Technologies
Installation
Usage
Model Information
Datasets
Deployment
Contributing

# Introduction
This project provides a web-based interface where users can input medical parameters and receive a prediction on the likelihood of having diabetes, heart disease, or Parkinson's disease. The predictions are made using machine learning models trained on respective datasets.

# Features
Disease Prediction: Provides prediction for the following diseases:
Diabetes
Heart Disease
Parkinson's Disease
Machine Learning Models: Utilizes SVM and Logistic Regression for predictions.
User Interface: Interactive and user-friendly web interface powered by Streamlit.

# Technologies
The following technologies and libraries are used in this project:

Python: Core programming language.
Streamlit: For creating the web interface.
scikit-learn: Machine learning library for model implementation.
NumPy: For numerical computations.
Pandas: For data manipulation and analysis.

# Installation
Prerequisites
Python 3.7+
Docker (optional for containerization)
Git for version control
Steps to Set Up the Project Locally

Clone the repository:
git clone https://github.com/yourusername/multiple_disease_prediction_model.git
cd multiple disease prediction model

Install the required Python packages:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Open your browser and navigate to http://localhost:8501.

Docker Setup (Optional)
If you prefer to use Docker for deployment, follow these steps:

Build the Docker image:
docker build -t disease-prediction-app .

Run the Docker container:
docker run -p 8501:8501 disease-prediction-app

# Usage
Once the app is running, you can enter the required medical parameters such as age, BMI, blood pressure, and other clinical information for each respective disease prediction. The app will return whether the inputted parameters indicate the presence of the selected disease.

# Model Information
The following machine learning models are used in this project:

Support Vector Machine (SVM): Used for high-dimensional space classification, effective in cases where the number of features is greater than the number of samples.
Logistic Regression: A linear model for binary classification, suitable for predicting binary outcomes.
Datasets
Diabetes: The model uses the PIMA Indian Diabetes dataset.
Heart Disease: Heart Disease dataset is used for predicting heart disease.
Parkinson's Disease: Parkinson's Disease dataset is used for prediction.
All datasets are publicly available.

# Deployment
The project can be deployed in a few ways:

Locally: Using Streamlit to run the app on your local machine.
Docker: The project is containerized using Docker, allowing easy deployment across platforms.
Cloud Deployment: You can also deploy the app on cloud platforms like AWS, GCP, or Azure using Docker.
For Vercel deployment, a vercel.json configuration is provided.

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug reports, or general improvements.
