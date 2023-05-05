import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import pickle


demo = pd.read_csv('demo.csv')

kmeans = KMeans(n_clusters = 5)
kmeans.fit_predict(demo)

with open('demographic_segmentation.pkl' , 'wb') as kmod:
    model = pickle.dump(kmeans,kmod)
