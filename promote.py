import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name='easyvisa_gbm',
    version=9,
    stage='Production',
    archive_existing_versions=True
)
print('Version 9 promoted to Production')
