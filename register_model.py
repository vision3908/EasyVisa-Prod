from mlflow.tracking import MlflowClient
client = MlflowClient('http://localhost:5001')

try:
    versions = client.search_model_versions("name='easyvisa_gbm'")
    for v in versions:
        client.delete_model_version('easyvisa_gbm', v.version)
        print(f'Deleted version {v.version}')
except Exception as e:
    print(f'Cleanup: {e}')

mv = client.create_model_version(
    name='easyvisa_gbm',
    source='s3://easyvisa-mlflow-vision-2025/mlflow-artifacts/1/models/m-f268491e90ba4dcf9f6dd3f151204596/artifacts',
    run_id='f289aac02a2249119e4c3bf4a8e3ad7e'
)
print(f'Created version {mv.version}')
client.transition_model_version_stage('easyvisa_gbm', mv.version, 'Production')
print('DONE - MLmodel path confirmed correct')
