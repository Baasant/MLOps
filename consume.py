import mlflow
from sklearn import datasets
iris=datasets.load_iris()
X=iris.data
client=mlflow.MlflowClient("sqlite:///my_db.db")
versions=client.search_model_versions(filter_string="name='iris_clustering_model'")

#get the latest version
latest_prod_version=client.get_latest_versions(name="iris_clustering_model",stages=["Production"])
model_uri=latest_prod_version[0].source


model=mlflow.pyfunc.load_model(model_uri)

#loading some test data
y_predict=model.predict(X[15:19])
print(y_predict)
