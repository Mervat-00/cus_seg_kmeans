import pickle 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

feed = pd.read_csv('feed.csv')

code = {}
cols = ['Marital_Status' , 'Education']
for col in cols:
    code[col] = LabelEncoder().fit(feed[col])
    
for col in cols :
    feed[col] = code[col].transform(feed[col])

scaler = StandardScaler()
scaler.fit(feed)
feed = pd.DataFrame(scaler.transform(feed))
feed.to_csv(r'C:\Users\kamel\OneDrive\Desktop\work\PROJECTS\customer segmentation using python\demographic k-means\meow.csv' , index=False )


with open('demographic_segmentation.pkl' , 'rb') as kmod:
    model = pickle.load(kmod)

predictions = model.predict(feed)
print(model.labels_)
