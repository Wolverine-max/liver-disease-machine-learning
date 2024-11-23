import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
import pickle 

st.title('Liver prediction App')

st.info('This app is for Liver Prediction on the basis of Medical Data! ')
#load model
@st.cache_resource
def load_model():
    model=pickle.dump(model, open("liver.pkl","rb"))
return model
