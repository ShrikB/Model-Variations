import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

#For Subset Selection for Preprocessing, FCVC, TUE, and MTRANS are dropped due to modern research
#of physical activity, diet fads, and accessory activities that un-intentionally skew bias
#More explanation in report
data_features = raw_data.drop(columns=['NObeyesdad', 'FCVC', 'TUE', 'MTRANS'])
data_target = raw_data['NObeyesdad'] #Unused

#Preprocessing Data
processed_target = pd.get_dummies(data_target)

#Sperating continous Data and scaling 
data_feat_cont = data_features[['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']]
data_feat_cont_scaled = pd.DataFrame(StandardScaler().fit_transform(data_feat_cont), columns=data_feat_cont.columns)

#Sperating categorical data and one hot encode
data_feat_cate = data_features[['Gender', 'CAEC', 'CALC']]
data_feat_cate_ohe = pd.get_dummies(data_feat_cate)

#Sperating variable data, updated as strings responding poorly
data_feat_var = data_features[['family_history_with_overweight', 'SMOKE', 'SCC']]
data_feat_var_proc = pd.get_dummies(data_feat_var)

#Combining all feature sets together
processed_features = pd.concat([data_feat_cont_scaled, data_feat_cate_ohe, data_feat_var_proc], axis=1)

#Utilizing KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(processed_features)
labels = kmeans.labels_

#To see how effective clustering is, seperated only Type 3 Obesity into a seperate dataset and used to predict placement
###################testing cluster###############
test_data = pd.concat([processed_features, processed_target], axis=1)
filtered_test_data = test_data[test_data['Obesity_Type_III'] == True]
filtered_test_data_features = filtered_test_data.drop(columns=processed_target.columns)
obesity_data = kmeans.predict(filtered_test_data_features)
###################testing cluster###############

#PCA Visualation for N Dimension dataset to 2D for plotting
pca = PCA(n_components=2)
processed_features_2d = pca.fit_transform(processed_features)
obes_feat_2d = pca.transform(filtered_test_data_features)

#Plotting Dataset
#Normal Dataset Cluster
unique_labels = set(labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    cluster_points = processed_features_2d[labels == label]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        c=[color],
        label=f"Cluster of Weight {label+1}",
        s=30,
        alpha=0.7,
        edgecolor="k",
        zorder=1 
    )
#Obese/Testing Cluster
plt.scatter(
    obes_feat_2d[:, 0],
    obes_feat_2d[:, 1],
    c=obesity_data,
    cmap="cool",
    s=50,  
    alpha=1.0,
    marker="x",
    label=f"Obesity Type 3 Test Set",
    linewidth=2, 
    zorder=3 
)

plt.title(f"Clustering with Overlay of Obesity Type 3 Test", fontsize=16)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
