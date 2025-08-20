import streamlit as slt
import pickle
import numpy as np
import pandas as pd

#import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

slt.title("Laptop Price Predictor")

#brand
company = slt.selectbox('Brand', df['Company'].unique())

#Type of laptop
type = slt.selectbox('Type', df['TypeName'].unique())

#Ram
ram = slt.selectbox('Ram(in GB)', [2,4,6,8,10,12,16,32,64])

# weight
weight = slt.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = slt.selectbox('Touchscreen', ['No','Yes'])

# IPS
ips = slt.selectbox('IPS', ['No','Yes'])

# screen size
screen_size = slt.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = slt.selectbox('Screen Resolution', [
    '1920x1080','1366x768','1600x900',
    '3840x2160','3200x1800','2880x1800',
    '2560x1600','2560x1440','2304x1440'
])

#cpu
cpu = slt.selectbox('CPU', df['Cpu brand'].unique())

hdd = slt.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

ssd = slt.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

gpu = slt.selectbox('GPU', df['Gpu brand'].unique())

os = slt.selectbox('OS', df['os'].unique())

if slt.button('Predict Price'):
    # Touchscreen and IPS encoding
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # ✅ Build dataframe instead of numpy array
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': ips,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Prediction
    prediction = np.exp(pipe.predict(query)[0])
    slt.title(f"Predicted Price: ₹{int(prediction)}")
