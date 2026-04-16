from mlflow.tracking import MlflowClient

client = MlflowClient('http://localhost:5001')

# Create model if not exists
try:
    client.create_registered_model('easyvisa_gbm')
    print('Model created')
except Exception as e:
    print(f'Model exists or error: {e}')

# Register correct version
mv = client.create_model_version(
    name='easyvisa_gbm',
    source='s3://easyvisa-mlflow-vision-2025/mlflow-artifacts/1/models/m-f268491e90ba4dcf9f6dd3f151204596/artifacts',
    run_id='f289aac02a2249119e4c3bf4a8e3ad7e'
)
print(f'Created version {mv.version}')
client.transition_model_version_stage('easyvisa_gbm', mv.version, 'Production')
print('DONE')
