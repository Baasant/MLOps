import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run():
    mlflow.log_param("my_parameter",1000)
