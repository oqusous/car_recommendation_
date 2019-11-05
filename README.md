# Car Recommender

## Data Source
Two websites were scraped using Beautiful Soup and Selenium for all models and trims of 43 Car brands - 2234 in total.

<figure>
<img src=images/trim_model.png title="Example of a Brand/Model/Trim" width="600"/>
  <figcaption>Fig1. - Example of a Brand/Model/Trim.</figcaption>
</figure>

## Feature Engineering
Physical: external and internal dimensions and weight of the car
Performance: Horsepower, Cylinders, Torque and Gear Ratios
Value: Price
Engine type: Fuel Capacity, Hybrid or Gas and miles/gallon
Total- 35 features
Domain knowledge selection
Continuous Features:
MinMaxScaler
Feature interaction: interior space and gear ratios
Categorical
Dummies for passenger capacity, engine type, number of cylinders, etc..
Missing data mainly handled using KNN Classifier and Regressor

<img src=images/gas_heatmap.png alt="Feature heatmap plot" width="600"/>
<img src=images/pairplots_gas.png alt="Feature pair plot" width="600"/>

## Modelling
Clustering:
KMeans
<img src=images/3d_plot1.png alt="3D PCA K Means plot" width="600"/>
KAgglomerativeClustering
<img src=images/3d_plot2.png alt="3D PCA K Agglomerative Clustering plot" width="600"/>
Annoy - Spotify made
Clusters, k, varied between 20-250; k=110 gave best results.
Ward linkage in KAgglomerativeClustering parameter outperformed the rest of the available methods.
Both were run with and without PCA. 
<img src=images/kmeans-silo.png alt="KMeans Silhouette Plot" width="600"/>
<img src=images/kh-silo.png alt="K Agglomerative Clustering Silhouette Plot" width="600"/>

|Model                                                |calinski_harabasz_score  | silhouette_score   |
|-----------------------------------------------------|-------------------------|--------------------|
|KMeans (k=30,)                                       | 642                     | 0.43               |
|KAgglomerativeClustering (k=110, linkage = single)   | 289                     | 0.54               |
|KAgglomerativeClustering (k=110,linkage = ward)      | 811                     | 0.55               |
|KAgglomerativeClustering (k=110,linkage = average)   | 428                     | 0.58               |
|KAgglomerativeClustering (k=110,linkage = complete)  | 475                     | 0.53               |
|PCA 15 Components with KAgglomerativeClustering      | 730                     | 0.48               |

