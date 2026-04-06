import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions('easyvisa_gbm', stages=['Production'])
v = versions[0]
print('Production version:', v.version)
print('Run ID:', v.run_id)
arts = client.list_artifacts(v.run_id, 'model')
for a in arts:
    print(' ', a.path)
