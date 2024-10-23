# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:35:32 2024

@author: gourav
"""

import numpy as np
import pickle 

loaded_model = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Heart Disease Prediction model/heart_disease_prediction_model.sav",'rb'))
scaler = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Heart Disease Prediction model/scaler.sav",'rb'))

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
#changing into numpy array then reshaping it and standardizing it 
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')