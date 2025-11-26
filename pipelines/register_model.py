import mlflow

run_id = dbutils.fs.head("/Volumes/main/gold/mlflow_volume/latest_run_id.txt")
model_uri = f"runs:/{run_id}/model"

registered_model_name = "fraud_pipeline_model"

mlflow.register_model(model_uri, registered_model_name)
