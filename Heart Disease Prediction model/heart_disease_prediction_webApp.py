# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:39:43 2024

@author: gourav
"""

import numpy as np 
import pickle 
import streamlit as st

loaded_model = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Heart Disease Prediction model/heart_disease_prediction_model.sav",'rb'))
scaler = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Heart Disease Prediction model/scaler.sav",'rb'))

def heartDisease_prediction(input_data):
    #changing into numpy array then reshaping it and standardizing it 
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    std_data = scaler.transform(input_data_reshaped)
    
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0]== 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'
  
def main():
    st.title('Heart Disease Prediction Web App')
    
    #getting user input 
    #age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
    Age = st.text_input('Age : ')
    col1,col2 = st.columns(2)
    
    with col1:
        Sex = st.text_input('Sex (Male-1, Female-0) : ')
        Cp = st.text_input('cp : ')
        Trestbps = st.text_input('trestbps : ')
        Chol = st.text_input('chol : ')
        Fbs = st.text_input('fbs : ')
        Restecg = st.text_input('restecg : ')
    
    with col2:
        Thalach = st.text_input('thalach : ')
        Exang = st.text_input('exang : ')
        Oldpeak = st.text_input('oldpeak : ')
        Slope = st.text_input('slope : ')
        Ca = st.text_input('ca : ')
        Thal = st.text_input('thal : ')
    
    diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        diagnosis = heartDisease_prediction([Age,Sex,Cp,Trestbps,Chol,Fbs,Restecg,Thalach,Exang,Oldpeak,Slope,Ca,Thal])
    
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    
    
    
    
    