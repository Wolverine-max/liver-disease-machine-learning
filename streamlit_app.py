import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title('Liver prediction App')

st.info('This app is for Liver Prediction on the basis of Medical Data! ')
#load model
@st.cache_resource
def scale_model():
    model=pickle.dump(model, open("liver.pkl","rb"))
@st.cache_resource
def load_scaling_params():
    with open('scaling_params.pkl', 'rb') as f:
        scaling_params = pickle.load(f)
    return scaling_params
    
def main():
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=0.8)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=100)
    sgpt = st.number_input("SGPT", min_value=0, value=20)
    sgot = st.number_input("SGOT", min_value=0, value=20)
    total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.0)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
    if st.button('Classify'):
        # Prepare the input data
        gender_val = 1 if gender == 'Male' else 0
        user_input = np.array([[age, gender_val, total_bilirubin, alkaline_phosphotase, sgpt, sgot, total_proteins, albumin]])
        scaling_params = load_scaling_params()
        train_mean = scaling_params['mean']
        train_std = scaling_params['std']
        # Load the pre-trained model
        model = load_model()
        
        # Make prediction
        prediction = model.predict(user_input_scaled)
        
        # Output the prediction
        if prediction == 1:
            st.error("The model predicts liver disease is present.")
        else:
            st.success("The model predicts no liver disease.")

if __name__ == '__main__':
    main()
   
