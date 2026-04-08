import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='easyvisa_gbm'")
for v in sorted(versions, key=lambda x: int(x.version)):
    print(f'Version {v.version} | Stage: {v.current_stage} | Run: {v.run_id[:12]}')
