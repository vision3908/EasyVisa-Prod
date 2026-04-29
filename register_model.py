import mlflow, sys
mlflow.set_tracking_uri('http://localhost:5001')
client = mlflow.tracking.MlflowClient()
with mlflow.start_run() as run:
    mlflow.log_param('source', 'restored-from-s3')
    run_id = run.info.run_id
    print('run_id:', run_id)
try:
    client.delete_registered_model('easyvisa_gbm')
except:
    pass
client.create_registered_model('easyvisa_gbm')
mv = client.create_model_version(
    'easyvisa_gbm',
    's3://easyvisa-mlflow-vision-2025/mlflow-artifacts/1/models/m-f516ecb85f7d405baca9efdac414171f/artifacts',
    run_id
)
client.transition_model_version_stage('easyvisa_gbm', mv.version, 'Production')
print('Done - version:', mv.version)
