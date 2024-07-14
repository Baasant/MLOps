#For auto log no need for log any parameters or mertix or plots

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mlflow
#generic
# mlflow.autolog()

#if you need more specific 
#here I use sklean
mlflow.set_tracking_uri("sqlite:///my_db.db")
mlflow.sklearn.autolog()

iris=datasets.load_iris()
x=iris.data
length=x.shape[0]
width=x.shape[1]


interias=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=100)
    kmeans.fit(x)
    interia=kmeans.inertia_
    interias.append(interia)
# Select the optimal k as the one with the least inertia
optimal_k = interias.index(min(interias)) + 2

#rerun the clustering with optimal k plot the distribution and log it as figure 
optimal_kmeans=KMeans(n_clusters=optimal_k,random_state=100)
labels=optimal_kmeans.fit_predict(x)
    # Plot the distribution and log it as a figure
plt.figure(figsize=(8, 6))
plt.hist(labels, bins=range(optimal_k + 1), align='left', rwidth=0.8)
plt.title(f'Iris Clustering with k={optimal_k}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
fig = plt.gcf()  # Get the current figure






     
