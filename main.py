import mlflow
# with mlflow.start_run(tags={"Algo":"Random Forest"}):
#      print("hello from tags")


#create new experiment
# mlflow.create_experiment(name="expermenting_regression2")

with mlflow.start_run(experiment_id="0"):
    print("hello from exp 0")

exp=mlflow.get_experiment(experiment_id="0")
print(exp.name)