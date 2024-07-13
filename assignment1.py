#Assignment 1
# 1-create  a main.py
# 2-create an experiment named iris_experiment 
# 3-create a run inside the experiment
#     3.1 log the shape of the dataframe in a dict
#         {"length":the length of df ,"width": the number of columns} ->shape.json
#     3.2 try clustering with k ranging from 2 to 10 
#         log the paramete interia as a metric
#         select the opimal k ass the one with the least inetria_  :log it as a parameter
#     3.3 rerun the clustering with optimal k 
#     3.4 plot the distribution and log it as figure 

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mlflow
iris=datasets.load_iris()
x=iris.data
length=x.shape[0]
width=x.shape[1]
#create an experiment named iris_experiment 
#create new experiment
try:
    mlflow.create_experiment(name="iris_experiment")
except mlflow.exceptions.MlflowException:
    print("already exist")

# create a run inside the experiment
with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("iris_experiment").experiment_id):
        mlflow.log_dict({"length":length,"width":width},"shape.json")
        #try clustering with k ranging from 2 to 10 log the parameter interia as a metric
        interias=[]
        for k in range(2,11):
            kmeans=KMeans(n_clusters=k,random_state=100)
            kmeans.fit(x)
            interia=kmeans.inertia_
            interias.append(interia)
            mlflow.log_metric(f"inertia_k_{k}", interia)
        # Select the optimal k as the one with the least inertia
        optimal_k = interias.index(min(interias)) + 2
        mlflow.log_param(key="opimal_k",value=optimal_k)

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
        mlflow.log_figure(fig, artifact_file="iris_clustering.png")






     
