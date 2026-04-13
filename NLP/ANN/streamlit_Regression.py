import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Load the trained model and scaler
model = tf.keras.models.load_model(os.path.join(script_dir, 'salary_regression_model.h5'))

#Load all the encoders and scalers
with open(os.path.join(script_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(script_dir, 'onehot_encoder_geo.pkl'), 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open(os.path.join(script_dir, 'label_encoder_gender.pkl'), 'rb') as f:
    label_encoder_gender = pickle.load(f)

# Streamlit app
st.title("Salary Prediction")

# Input fields for user data
input_data = {
    
    'CreditScore': st.slider("Credit Score", min_value=300, max_value=850, value=600),
	'Gender': st.selectbox("Gender", label_encoder_gender.classes_),
	'Age': st.number_input("Age", min_value=18, max_value=100, value=30),
    'Tenure': st.slider("Tenure", min_value=0, max_value=10, value=3),
    'Balance': st.number_input("Balance", min_value=0.0, value=10000.0),
    'NumOfProducts': st.slider("Number of Products", min_value=1, max_value=4, value=1),
    'HasCrCard': st.toggle("Has Credit Card", value=False),
    'IsActiveMember': st.selectbox("Is Active Member", options=[0, 1]),
    'Exited': st.selectbox("Exited", options=[0, 1]),
	'Geography': st.selectbox("Geography", onehot_encoder_geo.categories_[0])
       
}

if st.button("Predict Salary"):
    # Convert the input data into a DataFrame
    input_data_df = pd.DataFrame([input_data])
    
    # Encode the geographical feature using the loaded one-hot encoder
    geo_encoded = onehot_encoder_geo.transform(input_data_df[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Convert the gender categorical data into numerical data using the loaded label encoder
    input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])
    # Concatenate the original input data with the encoded geographical features
    input_data_df = pd.concat([input_data_df.drop('Geography', axis=1), geo_encoded_df], axis=1)
    # Scale the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data_df)
    # Predict the salary using the loaded model
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Salary: ${prediction[0][0]:.2f}")
