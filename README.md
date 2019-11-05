# Car Recommender

## Function
<p>
Using Spotify's Annoy Library, a content based car recommender is created. The user inputs a favourite car of theirs and the model gives recommendations of different car brands/models/trims based on the characteristics of the given car. 
</p>

## Data Source
<p>
Two websites were scraped using Beautiful Soup and Selenium for all models and trims of 43 Car brands - 2234 in total.
</p>
<figure>
<img src=images/trim_model.png title="Example of a Brand/Model/Trim" width="600"/>
  <figcaption>Fig1. - Example of a Brand/Model/Trim.</figcaption>
</figure>

## Feature Engineering
<p> 
  
  <ul type="disc">
  Feature list:
    <il> Physical: external and internal dimensions and weight of the car</il>
    <il>Performance: Horsepower, Cylinders, Torque and Gear Ratios</il>
    <il>Value: Price</il>
    <il>Engine type: Fuel Capacity, Hybrid or Gas and miles/gallon</il>
  </ul>
Total- 35 features
</p>
<br></br>
<p>
Features were selected based on domain knowledge.
<ol>Continuous Features were:
  <il>Scaled using MinMaxScaler.</il>
  <il>and for some, feature interaction was utilized eg. interior space and shift gear ratios.</il>
  <il>Missing data were mainly handled using KNN Regressor.</il>
</ol>
<ol> Categorical were:
  <il>Dummied for passenger capacity, engine type, number of cylinders, etc..</il>
  <il>Missing data were mainly handled using KNN Classifier.</il>
</ol>

<figure>
<img src=images/gas_heatmap.png alt="Feature heatmap plot" width="600"/>
  <figcaption>Fig2. - Feature heatmap plot.</figcaption>
</figure>
<br></br>
<figure>
<img src=images/pairplots_gas.png alt="Feature pair plot" width="600"/>
  <figcaption>Fig3. - Feature pair plot.</figcaption>
</figure>
<br></br>
## Modelling
All methods used were unsupervised nearest neighbour clustering algorithms:

1. sklearn's KMeans 

<figure>
<img src=images/3d_plot1.png alt="3D PCA K Means plot" width="600"/>
  <figcaption>Fig4. - 3D feature PCA K-Means plot.</figcaption>
</figure>

2. sklearn's KAgglomerativeClustering

<figure>
<img src=images/3d_plot2.png alt="3D PCA K Agglomerative Clustering plot" width="600"/>
  <figcaption>Fig5. - 3D feature PCA K Agglomerative Clustering plot.</figcaption>
</figure>

3. Annoy - Spotify inhouse library

Clusters, k, varied between 20-250; k=110 gave best results according to Silhouette plots below, and Calinski Harabasz scores.
Ward linkage in KAgglomerativeClustering parameter outperformed the rest of the available methods.
Both were run with and without PCA feature reduction.

<figure>
<img src=images/kmeans-silo.png alt="KMeans Silhouette Plot" width="600"/>
 <figcaption>Fig6. - KMeans Silhouette Plot.</figcaption>
</figure>

<figure>
<img src=images/kh-silo.png alt="K Agglomerative Clustering Silhouette Plot" width="600"/>
 <figcaption>Fig7. - K Agglomerative Clustering Silhouette Plot.</figcaption>
</figure>

Summary of Calinski Harabasz scores and Silhouette scores shown in the table below.

|Model                                                |calinski_harabasz_score  | silhouette_score   |
|-----------------------------------------------------|-------------------------|--------------------|
|KMeans (k=30,)                                       | 642                     | 0.43               |
|KAgglomerativeClustering (k=110, linkage = single)   | 289                     | 0.54               |
|KAgglomerativeClustering (k=110,linkage = ward)      | 811                     | 0.55               |
|KAgglomerativeClustering (k=110,linkage = average)   | 428                     | 0.58               |
|KAgglomerativeClustering (k=110,linkage = complete)  | 475                     | 0.53               |
|PCA 15 Components with KAgglomerativeClustering      | 730                     | 0.48               |
|(cummulative sum of explained variance ratio         |                         |                    |
|equaled 95%)                                         |                         |                    |

## Conclusion and Future Improvements
The Annoy model acheived best results and the algorithim was integrated into a front end streamlit script. This was determined through examining different examples of cars input into the Annoy vs KAgglomerativeClustering (k=110,linkage = complete) vs PCA 15 Components with KAgglomerativeClustering models.

Improvements:
● Integrate electric cars and 2020 models.
● Find other data sources to scrape for cars that required more KNN predictions.
● Hybrid recommender with user and expert ratings.
● Formulate value for options like infotainment system and interior material quality.

