import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title('Liver prediction App')

st.info('This app is for Liver Prediction on the basis of Medical Data! ')
#load model
@st.cache_resource
def load_model():
    try:
        with open('liver.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            return model
            
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None
        
@st.cache_resource
def load_scaling_params():
    try:
        with open('scaling_params.pkl', 'rb') as f:
            scaling_params = pickle.load(f)
        return scaling_params
        
     except Exception as e:
        st.error(f"Error loading scaling parameters: {e}")
        return None
    
def main():
    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=0.8)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=100)
    sgpt = st.number_input("SGPT", min_value=0, value=20)
    sgot = st.number_input("SGOT", min_value=0, value=20)
    total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.0)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
    bilirubin_direct = st.number_input("Direct Bilirubin", min_value=0.0, value=0.2) 
    alkaline_phosphotase_direct = st.number_input("Direct Alkaline Phosphotase", min_value=0, value=120)  
    if st.button('Predict'):
        # Prepare the input data
        gender_val = 1 if gender == 'Male' else 0
        user_input = np.array([[age, gender_val, total_bilirubin, alkaline_phosphotase, sgpt, sgot, total_proteins, albumin, bilirubin_direct, alkaline_phosphotase_direct]])
        scaling_params = load_scaling_params()
        if scaling_params is None:
            return
            
        train_mean = scaling_params['mean']
        train_std = scaling_params['std']
        train_mean = np.array(train_mean)
        train_std = np.array(train_std)
        
        if len(train_mean) != user_input.shape[1]:
            st.error(f"Mismatch in number of features: train_mean has {len(train_mean)} features, but user_input has {user_input.shape[1]} features.")
            return

        user_input_scaled = (user_input - train_mean) / train_std

        # Check that the scaled input is in the correct shape (2D array with 1 row and n features)
        if user_input_scaled.ndim == 1:
            user_input_scaled = user_input_scaled.reshape(1, -1)  # Reshape to 2D
             

        # Debugging: Check the shape of the input before prediction
        st.write(f"Scaled user input shape: {user_input_scaled.shape}")

        # Load the pre-trained model
        model = load_model()
        
        if model is None:
            st.error("Model failed to load. Please check the model file.")
            return
        
        # Make prediction
        try:
            prediction = model.predict(user_input_scaled)
            st.write(f"Prediction: {prediction}")
            
            # Output the prediction
            if prediction == 1:
                st.error("The model predicts liver disease is present.")
            else:
                st.success("The model predicts no liver disease.")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
if __name__ == '__main__':
    main()
   
