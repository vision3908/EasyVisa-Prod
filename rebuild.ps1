# rebuild.ps1 - EasyVisa V13 - PostgreSQL RDS Backend

$CLUSTER_NAME = "easyvisa-cluster-v3"
$REGION = "us-east-1"
$NAMESPACE = "easyvisa"
#$RDS_URI = "postgresql://mlflow:easyvisa123@easyvisa-mlflow-db.c6x82666ejqw.us-east-1.rds.amazonaws.com:5432/postgres"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " REBUILD - EasyVisa V13" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1 - Create EKS Cluster
Write-Host "`n[1/7] Creating EKS cluster..." -ForegroundColor Yellow
eksctl create cluster `
    --name $CLUSTER_NAME `
    --region $REGION `
    --nodegroup-name easyvisa-nodes `
    --node-type t3.medium `
    --nodes 3 `
    --nodes-min 2 `
    --nodes-max 4 `
    --managed

# Step 2 - Create namespace and secrets
Write-Host "`n[2/7] Creating namespace and secrets..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml

# Step 3 - Apply MLflow with RDS backend FIRST
Write-Host "`n[3/7] Deploying MLflow with RDS backend..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow-deployment.yaml

# Wait for MLflow to be ready
Write-Host "Waiting for MLflow to be ready..." -ForegroundColor Yellow
kubectl rollout status deployment/mlflow -n $NAMESPACE --timeout=300s

# Step 4 - Register model into RDS-backed MLflow
Write-Host "`n[4/7] Registering model into MLflow..." -ForegroundColor Yellow
Start-Job { kubectl port-forward -n easyvisa svc/mlflow 5000:5000 } | Out-Null
Start-Sleep -Seconds 10
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python src/train.py --data-path data/EasyVisa.csv
Get-Job | Stop-Job; Get-Job | Remove-Job

# Step 5 - Deploy all services
Write-Host "`n[5/7] Deploying all services..." -ForegroundColor Yellow
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/airflow-deployment.yaml

# Step 6 - Deploy Evidently
Write-Host "`n[6/7] Deploying Evidently..." -ForegroundColor Yellow
& .\deploy_evidently.ps1

# Step 7 - Show status
Write-Host "`n[7/7] Cluster status..." -ForegroundColor Yellow
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE

# Show public URLs
Write-Host "`n========================================" -ForegroundColor Green
Write-Host " PUBLIC SERVICE URLs:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
$API_URL = kubectl get svc visa-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
$GRAFANA_URL = kubectl get svc grafana-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
$AIRFLOW_URL = kubectl get svc airflow-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
Write-Host " API:     http://$API_URL/docs" -ForegroundColor Green
Write-Host " Grafana: http://$GRAFANA_URL (admin/easyvisa123)" -ForegroundColor  Green
#Write-Host " Airflow: http://$AIRFLOW_URL:8080 (admin/easyvisa123)" -ForegroundColor Green
Write-Host " Airflow: http://$($AIRFLOW_URL):8080 (admin/easyvisa123)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host " REBUILD COMPLETE - EasyVisa V13" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green