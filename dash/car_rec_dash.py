import pandas as pd
import numpy as np
import re
import time
import pickle as pkl
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from annoy import AnnoyIndex
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go

df_for_brands = pkl.load(open('df_pickles/df_reg_1.p','rb'))
df_for_brands.drop('price',1,inplace=True)
df_for_brands_gas = df_for_brands[df_for_brands['Engine type']!='Electric']
df_for_brands_elec = df_for_brands[df_for_brands['Engine type']=='Electric']
prices = pkl.load(open('df_pickles/prices.p','rb'))
df_for_brands_gas = pd.concat([df_for_brands_gas, prices],1)
df_gas = pkl.load(open('df_pickles/df_gas_sc_dmd.p','rb'))
df_gas_mod = df_gas.drop(['model', 'front_suspension_type_1',
       'front_suspension_type_2', 'front_suspension_type_3',
       'front_suspension_type_4', 'front_suspension_type_5',
       'front_suspension_type_6', 'front_suspension_type_7',
       'front_suspension_type_8', 'front_suspension_type_9', 
        'front_suspension_type_10', 'front_suspension_type_11',
       'front_suspension_type_12', 'front_wheel_diameter_15',
       'front_wheel_diameter_16', 'front_wheel_diameter_17',
       'front_wheel_diameter_18', 'front_wheel_diameter_19',
       'front_wheel_diameter_20', 'front_wheel_diameter_21',
       'front_wheel_diameter_22', 'rear_suspension_type_1',
       'rear_suspension_type_2', 'rear_suspension_type_3',
       'rear_suspension_type_4', 'rear_suspension_type_5', 
                          'rear_suspension_type_6',
       'rear_suspension_type_7', 'rear_suspension_type_8',
       'rear_suspension_type_9', 'rear_suspension_type_10',
       'rear_wheel_diameter_15', 'rear_wheel_diameter_16',
       'rear_wheel_diameter_17', 'rear_wheel_diameter_18',
       'rear_wheel_diameter_19', 'rear_wheel_diameter_20',
       'rear_wheel_diameter_21', 'rear_wheel_diameter_22',
                         'brand_1',
       'brand_2', 'brand_3', 'brand_4', 'brand_5', 'brand_6', 'brand_7',
       'brand_8', 'brand_9', 'brand_10', 'brand_11', 'brand_12', 'brand_13',
       'brand_14', 'brand_15', 'brand_16', 'brand_17', 'brand_18', 'brand_19',
       'brand_20', 'brand_21', 'brand_22', 'brand_23', 'brand_24', 'brand_25',
       'brand_26', 'brand_27', 'brand_28', 'brand_29', 'brand_30', 'brand_31',
       'brand_32', 'brand_33', 'brand_34', 'brand_35', 'brand_36', 'brand_37',
       'brand_39', 'brand_40', 'brand_41'], 1)
df_gas_mod['index'] = list(df_gas_mod.index)
df_for_brands_gas['index'] = list(df_for_brands_gas.index)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=3).fit_transform(df_gas_mod.iloc[:,:-1])
df_plot = pd.DataFrame(X_embedded, index=df_gas_mod.index, columns = ['x','y','z'])
agg_clust_ward_tsne = AgglomerativeClustering(linkage='ward', n_clusters=30)
assigned_clust_ward_tsne = agg_clust_ward_tsne.fit_predict(df_plot)
df_plot['labels']=assigned_clust_ward_tsne

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1('Car Recommender'),
    
    dcc.Markdown('''Enter your favorite car brand and model to get other cars with similar specs.'''),
    
    dcc.Markdown('Car brand'),
    dcc.Input(id='input-1-state', type='text', value=''),
    
    dcc.Markdown('Model'),
    
    dcc.Input(id='input-2-state', type='text', value=''),
    
    html.Button(id='submit-1-button', n_clicks=0, children='Submit'),
    
    dcc.Graph(
        id='basic-interactions',
        style={"height":850 , "width" : 1000, },
        figure={'data': [{
                    'x': df_plot['x'],
                    'y': df_plot['y'],
                    'z': df_plot['z'],
                     'mode':'markers',
                    'type':'scatter3d',
                    'text': df_plot.index,
            'marker':{'size': 8, 'color':df_plot['labels'], 'colorscale': 'Blackbody', 'opacity': 0.8},
            'layout': {'clickmode': 'event+select', 'height':1200, 
                       'title':'3D Diagram of t-distributed Stochastic Neighbor Embedding with three components'}}]}),
    
    html.Div(id='output-1-state'),
    dcc.Markdown('Paste Index Here'),
    dcc.Input(id='input-3-state', type='text', value=''),
    html.Button(id='submit-2-button', n_clicks=0, children='Submit'),
    html.Div(id='output-2-state')
])

def generate_table(dataframe, max_rows=26):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

svd = TruncatedSVD(n_components=11)
df_Annoy_svd = svd.fit_transform(df_gas_mod.iloc[:,:-1])
from annoy import AnnoyIndex
import random

f = df_Annoy_svd.shape[1] # Length of item vector that will be indexed
t = AnnoyIndex(f, metric='angular')  
for i in range(df_Annoy_svd.shape[0]):
    v = df_Annoy_svd[i]
    t.add_item(i, v)

t.build(15)
# t.save('test.ann')


@app.callback(Output('output-1-state', 'children'),
              [Input('submit-1-button', 'n_clicks')],
              [State('input-1-state', 'value'),
               State('input-2-state', 'value')])

def update_output1(n_clicks, input1, input2):
    print_output1 = True
    index_list = []
    for i in df_for_brands_gas.index:
        if re.findall(str(input1).lower()+'\/'+str(input2).lower(), i):
            index_list.append(i)
    if print_output1 == True:
        df = df_for_brands_gas.loc[index_list, ['brand','model','Torque','Passenger Capacity', 'price', 'trim', 'index']]
    return generate_table(df, max_rows=20)


@app.callback(Output('output-2-state', 'children'),
              [Input('submit-2-button', 'n_clicks')],
              [State('input-3-state', 'value')])

def update_output2(n_clicks, input3):
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
    return generate_table(df, max_rows=25)

if __name__ == '__main__':
    app.run_server(debug=False)
