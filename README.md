# Car Recommender
All models and trims of 43 Car brands - 2234 in total

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

## Modelling
Clustering:
KMeans
KAgglomerativeClustering
Annoy - Spotify made

Clusters, k, varied between 20-250; k=110 gave best results.
Ward linkage in KAgglomerativeClustering parameter outperformed the rest of the available methods.
Both were run with and without PCA. 

|Model                                                |calinski_harabasz_score  | silhouette_score  |
|-----------------------------------------------------|-------------------------|-------------------|
|KMeans (k=30,)                                       | 642                     |                   |
|KAgglomerativeClustering (k=110, linkage = single)   | 289                     |                   |
|KAgglomerativeClustering (k=110,linkage = ward)      | 811                     |                   |
|KAgglomerativeClustering (k=110,linkage = average)   | 428                     |                   |
|KAgglomerativeClustering (k=110,linkage = complete)  | 475                     |                   |
|PCA 15 Components with KAgglomerativeClustering      | 730                     |                   |

