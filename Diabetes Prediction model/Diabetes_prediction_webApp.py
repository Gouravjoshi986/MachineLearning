# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:54:10 2024

@author: gourav
"""

import numpy as np 
import pickle 
import streamlit as st

#loading the model 
loaded_model = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Diabetes Prediction model/diabetes_prediction_model.sav", 'rb'))
scaler = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Diabetes Prediction model/scaler.sav",'rb'))
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    #standardize input
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


def main():
    #giving title 
    st.title('Diabetes Prediction Web App')
    
    #getting user input 
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input('No of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree function value')
    Age = st.text_input('Age of Person')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Results'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    