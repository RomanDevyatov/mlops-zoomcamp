import mlflow

mlflow.set_experiment("test-experiment")

with mlflow.start_run():
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.25)
