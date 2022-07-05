

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:15:01 2021

@author: siddhardhan
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/Users/Gaoiz/OneDrive/Bureau/Master Data/DEPLOYMENT/trained_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is cancer benign '
    else:
      return 'The person is cancer malign'
  
    
  
def main():
    
    
    # giving a title
    st.title('Cancer Prediction Web App')
    
    
    # getting the input data from the user
    #texture_se,smoothness_se,symmetry_se,fractal_dimension_se,texture_worst,concave points_worst,symmetry_worst,fractal_dimension_worst
    
    texture_se = st.text_input('Number of Pregnancies')
    smoothness_se = st.text_input('Glucose Level')
    symmetry_se = st.text_input('Blood Pressure value')
    fractal_dimension_se = st.text_input('Skin Thickness value')
    texture_worst = st.text_input('Insulin Level')
    concave_points_worst = st.text_input('BMI value')
    symmetry_worst= st.text_input('Diabetes Pedigree Function value')
    fractal_dimension_worst= st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Cancer Test Result'):
        diagnosis = diabetes_prediction([texture_se,smoothness_se,symmetry_se,fractal_dimension_se,texture_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
