# -*- coding: utf-8 -*-

import numpy as np 
import pickle 

#loading the model 
loaded_model = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Diabetes Prediction model/diabetes_prediction_model.sav", 'rb'))
scaler = pickle.load(open("C:/Users/gourav/Desktop/PROGRAMMING LANGUAGES/machine learning/Diabetes Prediction model/scaler.sav",'rb'))
input_data = (5,166,72,19,175,25.8,0.587,51)

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
  print('The person is not diabetic')
else:
  print('The person is diabetic')