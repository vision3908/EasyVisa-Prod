import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
for v in client.search_model_versions("name='easyvisa_gbm'"):
    print(f"Version {v.version} | Stage: {v.current_stage} | Run: {v.run_id[:12]}")
