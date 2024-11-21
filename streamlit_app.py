import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd

st.title('Liver prediction App')

st.info('This app is for Liver Prediction on the basis of Medical Data! ')
df=pd.read_csv('https://github.com/Wolverine-max/liver-disease-machine-learning/blob/master/indian_liver_patient.csv')
df
