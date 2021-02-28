import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
from annoy import AnnoyIndex
import random

df_gas_mod = pkl.load(open('df_pickles/df_gas_mod_streamlite.p', 'rb'))
df_for_brands_gas = pkl.load(open('df_pickles/df_for_brands_gas_streamlite.p', 'rb'))
df_plot = pkl.load(open('df_pickles/df_plot_streamlite.p', 'rb'))
df_plot['index'] = list(df_plot.index)
df_plot.rename(columns={'x':'Dimension 1', 'y':'Dimension 2', 'z':'Dimension 3'}, inplace=True)

image = Image.open('images/enzo_resized.jpg')
st.image(image, caption='', use_column_width=False)
st.title('Car Recommender')

st.write('''Enter your favorite car brand and model and get car recommendations.''')

option1 = st.selectbox("Select car brand",options = list(set(df_for_brands_gas['brand'])))
option1a = option1

st.markdown('\n')

option2 = st.selectbox("Select the model",options= list(set(df_for_brands_gas[df_for_brands_gas['brand']==option1a]['model'])))
first= option1a.lower()
sec= option2.lower()

st.markdown('\n')

fig = px.scatter_3d(df_plot, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='labels',  title="""Three Component SVD Transformed Plot of Car Recommender Features Clustered with KAlgg""", height=800, width=800 ,hover_data=df_plot[['index']])
st.write('''Key: Naming convention of labels = car/year/make/model/model idx number''')
st.write(fig)

st.markdown('\n')

svd = TruncatedSVD(n_components=11)
df_Annoy_svd = svd.fit_transform(df_gas_mod.iloc[:,:-1])
f = df_Annoy_svd.shape[1]
t = AnnoyIndex(f, metric = 'angular')  
for i in range(df_Annoy_svd.shape[0]):
    v = df_Annoy_svd[i]
    t.add_item(i, v)
t.build(15)
# t.save('test.ann')

st.write(df_for_brands_gas[(df_for_brands_gas['brand']==option1) & (df_for_brands_gas['model']==option2)][['brand','model','Torque','Passenger Capacity', 'price', 'trim']])

index = df_for_brands_gas[(df_for_brands_gas['brand']==option1) & (df_for_brands_gas['model']==option2)][['brand','model','Torque','Passenger Capacity', 'price', 'trim']]

st.markdown('\n')

option3 = st.selectbox("Choose trim from list above to get recommendations on car with similar characteristics",options= list(index.index))
option3a = option3

st.markdown('\n')

def update_output2(input3):
    n=25
    print_output=True
    index = t
    val3_iloc = list(df_gas_mod.index).index(input3)
    nn = index.get_nns_by_item(val3_iloc, n)
    if print_output == True:
        print('Closest to %s : \n' % df_gas_mod.index[val3_iloc])
        cars = [df_gas_mod.index[i] for i in nn]
    if print_output == True:
        df = df_for_brands_gas.loc[cars, ['brand','model','Torque','Passenger Capacity', 'price', 'trim', 'index']]
        print(cars)
    return df.iloc[1:,:]

nns = update_output2(option3a)
st.write(nns)

st.markdown('\n')
