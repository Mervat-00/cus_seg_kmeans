import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def encode_columns(data, cols):
    code = {}
    for col in cols:
        code[col] = LabelEncoder().fit(data[col])
    for col in cols:
        data[col] = code[col].transform(data[col])
    return data

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data) 
    return pd.DataFrame(scaler.transform(data))

def preprocess_data(input_data):
    # Encode columns
    cols_to_encode = ['Marital_Status', 'Education']
    encoded_data = encode_columns(input_data, cols_to_encode)

    # Scale data
    norm_data = scale_data(encoded_data)

    return norm_data
