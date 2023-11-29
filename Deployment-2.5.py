#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle as pkl
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the pickled model
with open('deploylr.plk', 'rb') as file:
    lr = pkl.load(file)

# Streamlit app
st.title('Salary Prediction App')

# User input for Marks and Age
marks = st.slider('Marks', min_value=0, max_value=100, step=1)
age = st.slider('Age', min_value=20, max_value=60, step=1)

# Make a prediction using the model
prediction = lr.predict([[marks, age]])

# Display the prediction
st.write(f"Predicted Salary: {prediction[0]:.2f}")

# Additional features like displaying the model coefficients
st.write("Model Coefficients:")
st.write(f"Coefficients: {lr.coef_}")
st.write(f"Intercept: {lr.intercept_}")


# In[ ]:




