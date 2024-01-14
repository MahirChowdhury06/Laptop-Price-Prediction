import streamlit as st
#import sklearn
import pickle
#import numpy as np

import sys
print("Python version:", sys.version)
import numpy as np
print("NumPy version:", np.__version__)
import pandas as pd
print("Pandas version:", pd.__version__)
import xgboost as xgb
print("xgboost version:", xgb.__version__)
import sklearn
print("sklearn version:", sklearn.__version__)

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Brand'].unique())

# type of laptop
type = st.selectbox('Type',df['Type'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# OS
os = st.selectbox('Operating System',df['OpSys'].unique())

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,os,weight,touchscreen,ips,ppi,cpu,gpu])

    query = query.reshape(1,10)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

