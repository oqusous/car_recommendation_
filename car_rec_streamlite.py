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
from selenium import webdriver
from annoy import AnnoyIndex
import random
from selenium.webdriver.chrome.options import Options
# df_for_brands = pkl.load(open('df_pickles/df_reg_1.p','rb'))
# df_for_brands.drop('price',1,inplace=True)
# df_for_brands_gas = df_for_brands[df_for_brands['Engine type']!='Electric']
# df_for_brands_elec = df_for_brands[df_for_brands['Engine type']=='Electric']
# prices = pkl.load(open('df_pickles/prices.p','rb'))
# df_for_brands_gas = pd.concat([df_for_brands_gas, prices],1)
# df_gas = pkl.load(open('df_pickles/df_gas_sc_dmd.p','rb'))
# df_gas_mod = df_gas.drop(['model', 'front_suspension_type_1', 'front_suspension_type_2', 'front_suspension_type_3', 'front_suspension_type_4', 'front_suspension_type_5',
#        'front_suspension_type_6', 'front_suspension_type_7', 'front_suspension_type_8', 'front_suspension_type_9', 'front_suspension_type_10', 'front_suspension_type_11',
#        'front_suspension_type_12', 'front_wheel_diameter_15', 'front_wheel_diameter_16', 'front_wheel_diameter_17', 'front_wheel_diameter_18', 'front_wheel_diameter_19',
#        'front_wheel_diameter_20', 'front_wheel_diameter_21', 'front_wheel_diameter_22', 'rear_suspension_type_1', 'rear_suspension_type_2', 'rear_suspension_type_3',
#        'rear_suspension_type_4', 'rear_suspension_type_5', 'rear_suspension_type_6', 'rear_suspension_type_7', 'rear_suspension_type_8','rear_suspension_type_9', 'rear_suspension_type_10',
#        'rear_wheel_diameter_15', 'rear_wheel_diameter_16', 'rear_wheel_diameter_17', 'rear_wheel_diameter_18', 'rear_wheel_diameter_19', 'rear_wheel_diameter_20',
#        'rear_wheel_diameter_21', 'rear_wheel_diameter_22', 'brand_1', 'brand_2', 'brand_3', 'brand_4', 'brand_5', 'brand_6', 'brand_7', 'brand_8', 'brand_9', 'brand_10', 'brand_11', 'brand_12', 'brand_13',
#        'brand_14', 'brand_15', 'brand_16', 'brand_17', 'brand_18', 'brand_19', 'brand_20', 'brand_21', 'brand_22', 'brand_23', 'brand_24', 'brand_25',
#        'brand_26', 'brand_27', 'brand_28', 'brand_29', 'brand_30', 'brand_31', 'brand_32', 'brand_33', 'brand_34', 'brand_35', 'brand_36', 'brand_37', 'brand_39', 'brand_40', 'brand_41'], 1)

# df_gas_mod['index'] = list(df_gas_mod.index)
# df_for_brands_gas['index'] = list(df_for_brands_gas.index)
# pkl.dump(df_gas_mod, open('df_pickles/df_gas_mod_streamlite.p', 'wb'))
# pkl.dump(df_for_brands_gas, open('df_pickles/df_for_brands_gas_streamlite.p', 'wb'))

# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=3).fit_transform(df_gas_mod.iloc[:,:-1])
# df_plot = pd.DataFrame(X_embedded, index=df_gas_mod.index, columns = ['x','y','z'])
# agg_clust_ward_tsne = AgglomerativeClustering(linkage='ward', n_clusters=110)
# assigned_clust_ward_tsne = agg_clust_ward_tsne.fit_predict(df_plot)
# df_plot['labels']=assigned_clust_ward_tsne
# pkl.dump(df_plot, open('df_pickles/df_plot_streamlite.p', 'wb'))

df_gas_mod = pkl.load(open('df_pickles/df_gas_mod_streamlite.p', 'rb'))
df_for_brands_gas = pkl.load(open('df_pickles/df_for_brands_gas_streamlite.p', 'rb'))
df_plot = pkl.load(open('df_pickles/df_plot_streamlite.p', 'rb'))
df_plot['index'] = list(df_plot.index)
df_plot.rename(columns={'x':'Dimension 1', 'y':'Dimension 2', 'z':'Dimension 3'}, inplace=True)

image = Image.open('images/enzo.jpg')
st.image(image, caption='', use_column_width=False)
st.title('Car Recommender')

st.write('''Enter your favorite car brand and model and get car recommendations.''')

option1 = st.selectbox("Select car brand",options = list(set(df_for_brands_gas['brand'])))
option1a = option1

st.markdown('\n')

option2 = st.selectbox("Select the model",options= list(set(df_for_brands_gas[df_for_brands_gas['brand']==option1a]['model'])))
first= option1a.lower()
sec= option2.lower()
# options = Options()
# options.add_argument('--headless')
# options.add_argument('--disable-gpu')
# driver = webdriver.Chrome('/Users/flatironschool/Desktop/chromedriver')
# car_values='http://www.kbb.com/{}/{}'.format(first, sec)
# driver.get(car_values)
# try:
#     url = driver.find_element_by_css_selector('section[data-analytics="overview"]').find_element_by_tag_name('img').get_attribute('src')
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     st.image(img, caption='', use_column_width=False)
# except:
#     st.markdown('Sorry, no image available')
#     pass

st.markdown('\n')

fig = px.scatter_3d(df_plot, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='labels',  title="""Three Component SVD Transformed Plot of Car Recommender Features Clustered with KAlgg""", height=1000, width=1000 ,hover_data=df_plot[['index']])
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

# first1= st.selectbox("Car brand",options = list(set(nns['brand'])))
# first1a=first1
# sec1= st.selectbox("Model",options= list(set(nns[nns['brand']==first1a]['model'])))
# car2='http://www.kbb.com/{}/{}'.format(first1.lower(), sec1.lower())
# driver.get(car2)
# try:
#     url = driver.find_element_by_css_selector('section[data-analytics="overview"]').find_element_by_tag_name('img').get_attribute('src')
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     st.image(img, caption='', use_column_width=False)
# except:
#     st.markdown('Sorry, no image available')
#     pass
# driver.close()