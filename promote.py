import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='easyvisa_gbm'")
latest = sorted(versions, key=lambda x: int(x.version))[-1]
print('Promoting version:', latest.version, 'run:', latest.run_id)
client.transition_model_version_stage(
    name='easyvisa_gbm',
    version=int(latest.version),
    stage='Production',
    archive_existing_versions=True
)
print('Done!')
