import os
from mlflow.tracking import MlflowClient

tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
client = MlflowClient(tracking_uri)
print(f'Connecting to MLflow at: {tracking_uri}')

# Create model if not exists
try:
    client.create_registered_model('easyvisa_gbm')
    print('Model created')
except Exception as e:
    print(f'Model exists or error: {e}')

# Register correct version
mv = client.create_model_version(
    name='easyvisa_gbm',
    source='s3://easyvisa-mlflow-vision-2025/mlflow-artifacts/1/models/m-f289aac02a2249119e4c3bf4a8e3ad7e/artifacts/model',
    run_id='f289aac02a2249119e4c3bf4a8e3ad7e'
)
print(f'Created version {mv.version}')
client.transition_model_version_stage('easyvisa_gbm', mv.version, 'Production')
print('DONE')