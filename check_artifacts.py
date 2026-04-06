import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='easyvisa_gbm'")
latest = sorted(versions, key=lambda x: int(x.version))[-1]
print('Latest version:', latest.version, '| Run ID:', latest.run_id)
arts = client.list_artifacts(latest.run_id, 'model')
for a in arts:
    print(' ', a.path)
