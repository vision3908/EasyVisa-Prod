# =============================================================================
# rebuild.ps1 - EasyVisa-Prod Full Cluster Rebuild (V12 - Bulletproof)
#
# FIXES OVER V11:
#   1. OIDC provider associated BEFORE EBS CSI addon
#   2. IAM role created via eksctl (no more CSI CrashLoopBackOff)
#   3. EBS CSI addon with --service-account-role-arn
#   4. kubectl rollout status waits for MLflow READY before model registration
#   5. register_model.py called with MLFLOW_TRACKING_URI env var
#   6. visa-api restarted AFTER model registration
#   7. Airflow DAG + CSV copied post-deploy
#
# USAGE:  .\rebuild.ps1
# AFTER:  .\deploy_evidently.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$CLUSTER_NAME = "easyvisa-cluster-v3"
$REGION = "us-east-1"
$NAMESPACE = "easyvisa"
$ACCOUNT_ID = "167963468596"
$IAM_ROLE = "AmazonEKS_EBS_CSI_DriverRole"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " EasyVisa-Prod Rebuild Script V12" -ForegroundColor Cyan
Write-Host " Cluster: $CLUSTER_NAME | Region: $REGION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# STEP 0: Load .env
Write-Host "`n[0/9] Loading .env credentials..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) { Write-Error ".env not found. Aborting."; exit 1 }
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=\s]+)=(.+)$') {
        $k = $matches[1].Trim()
        $v = $matches[2].Trim()
        if ($k -notmatch "AIRFLOW_HOME|PYTHONPATH") {
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
}
if (-not $env:AWS_ACCESS_KEY_ID) { Write-Error "AWS_ACCESS_KEY_ID missing from .env. Aborting."; exit 1 }
Write-Host "   OK - Credentials loaded: $($env:AWS_ACCESS_KEY_ID.Substring(0,8))***" -ForegroundColor Green

# STEP 1: Create EKS Cluster
Write-Host "`n[1/9] Creating EKS cluster '$CLUSTER_NAME'..." -ForegroundColor Yellow
eksctl create cluster --name $CLUSTER_NAME --region $REGION --nodegroup-name easyvisa-nodes --node-type t3.medium --nodes 2 --nodes-min 2 --nodes-max 3 --managed
Write-Host "   OK - Cluster created." -ForegroundColor Green

# STEP 2: OIDC Provider
Write-Host "`n[2/9] Associating OIDC provider..." -ForegroundColor Yellow
eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --region $REGION --approve
Write-Host "   OK - OIDC provider associated." -ForegroundColor Green

# STEP 3: IAM Role for EBS CSI
Write-Host "`n[3/9] Creating IAM role for EBS CSI driver..." -ForegroundColor Yellow
$roleArn = aws iam get-role --role-name $IAM_ROLE --query "Role.Arn" --output text 2>$null
if ($roleArn -and $roleArn -notmatch "NoSuchEntity" -and $roleArn -notmatch "error") {
    Write-Host "   OK - IAM role already exists: $roleArn" -ForegroundColor Green
} else {
    eksctl create iamserviceaccount --name ebs-csi-controller-sa --namespace kube-system --cluster $CLUSTER_NAME --role-name $IAM_ROLE --role-only --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy --region $REGION --approve
    Write-Host "   OK - IAM role $IAM_ROLE created." -ForegroundColor Green
}

# STEP 4: kubeconfig + EBS CSI Addon
Write-Host "`n[4/9] Updating kubeconfig and installing EBS CSI addon..." -ForegroundColor Yellow
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

$ebsRoleArn = "arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE}"
eksctl create addon --name aws-ebs-csi-driver --cluster $CLUSTER_NAME --region $REGION --service-account-role-arn $ebsRoleArn --force
Write-Host "   OK - EBS CSI addon created." -ForegroundColor Green

Write-Host "   Waiting for EBS CSI addon ACTIVE (up to 5 min)..." -ForegroundColor White
$addonReady = $false
for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 15
    $addonOut = eksctl get addon --name aws-ebs-csi-driver --cluster $CLUSTER_NAME --region $REGION 2>$null
    if ($addonOut -match "ACTIVE") {
        Write-Host "   OK - EBS CSI addon is ACTIVE." -ForegroundColor Green
        $addonReady = $true
        break
    }
    Write-Host "   Still waiting... ($($i * 15)s elapsed)" -ForegroundColor White
}
if (-not $addonReady) {
    Write-Host "   WARN - EBS CSI not ACTIVE after 5 min. Check manually." -ForegroundColor Yellow
}

# STEP 5: Namespace + Secret
Write-Host "`n[5/9] Creating namespace and secrets..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml

kubectl create secret generic easyvisa-secrets --namespace=easyvisa --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" --from-literal=aws-default-region="us-east-1" --from-literal=AWS_ACCESS_KEY_ID="$env:AWS_ACCESS_KEY_ID" --from-literal=AWS_SECRET_ACCESS_KEY="$env:AWS_SECRET_ACCESS_KEY" --from-literal=AWS_DEFAULT_REGION="us-east-1" --from-literal=MLFLOW_TRACKING_URI="http://mlflow:5000" --from-literal=MLFLOW_MODEL_URI="models:/easyvisa_gbm/Production" --from-literal=mlflow-tracking-uri="http://mlflow:5000" --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production" --dry-run=client -o yaml | kubectl apply -f -
Write-Host "   OK - Namespace and secrets ready." -ForegroundColor Green

# STEP 6: Deploy All Services
Write-Host "`n[6/9] Deploying all services to EKS..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow-pvc.yaml --validate=false
kubectl apply -f k8s/mlflow-deployment.yaml --validate=false
kubectl apply -f k8s/deployment.yaml --validate=false
kubectl apply -f k8s/service.yaml --validate=false
kubectl apply -f k8s/prometheus-deployment.yaml --validate=false
kubectl apply -f k8s/grafana-deployment.yaml --validate=false
kubectl apply -f k8s/airflow-deployment.yaml --validate=false
Write-Host "   OK - All manifests applied." -ForegroundColor Green

# STEP 7: Wait for MLflow, register model, restart visa-api
Write-Host "`n[7/9] Waiting for MLflow to be ready..." -ForegroundColor Yellow
kubectl rollout status deployment/mlflow -n $NAMESPACE --timeout=180s
Write-Host "   OK - MLflow rollout complete." -ForegroundColor Green
Start-Sleep -Seconds 20

Write-Host "   Starting MLflow port-forward for model registration..." -ForegroundColor White
$pfJob = Start-Job -ScriptBlock { kubectl port-forward -n easyvisa svc/mlflow 5000:5000 }
Start-Sleep -Seconds 10

$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python register_model.py
Write-Host "   OK - Model registered in cluster MLflow." -ForegroundColor Green

Stop-Job $pfJob -ErrorAction SilentlyContinue
Remove-Job $pfJob -ErrorAction SilentlyContinue
[System.Environment]::SetEnvironmentVariable("MLFLOW_TRACKING_URI", $null, "Process")

Start-Sleep -Seconds 5
kubectl rollout restart deployment/visa-api-deployment -n $NAMESPACE
Write-Host "   OK - visa-api restarted after model registration." -ForegroundColor Green

# STEP 8: Copy Airflow DAG + CSV
Write-Host "`n[8/9] Copying DAG and CSV into Airflow pod..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
$AF_POD = kubectl get pods -n $NAMESPACE -l app=airflow -o jsonpath='{.items[0].metadata.name}' 2>$null
if ($AF_POD) {
    kubectl cp airflow/dags/easyvisa_retrain_dag.py "${NAMESPACE}/${AF_POD}:/opt/airflow/dags/easyvisa_retrain_dag.py"
    kubectl cp data/EasyVisa.csv "${NAMESPACE}/${AF_POD}:/opt/airflow/data/EasyVisa.csv"
    kubectl exec -n $NAMESPACE $AF_POD -- airflow db init 2>$null
    kubectl exec -n $NAMESPACE $AF_POD -- airflow users create --username admin --password easyvisa123 --firstname E --lastname V --role Admin --email admin@easyvisa.local 2>$null
    Write-Host "   OK - Airflow DAG + CSV copied. Login: admin / easyvisa123" -ForegroundColor Green
} else {
    Write-Host "   WARN - Airflow pod not ready. Copy manually:" -ForegroundColor Yellow
    Write-Host '   $AF_POD = kubectl get pods -n easyvisa -l app=airflow -o jsonpath=''{.items[0].metadata.name}''' -ForegroundColor White
    Write-Host "   kubectl cp airflow/dags/easyvisa_retrain_dag.py easyvisa/`$AF_POD:/opt/airflow/dags/" -ForegroundColor White
    Write-Host "   kubectl cp data/EasyVisa.csv easyvisa/`$AF_POD:/opt/airflow/data/EasyVisa.csv" -ForegroundColor White
}

# STEP 9: Final Status + URLs
Write-Host "`n[9/9] Waiting 60s for LoadBalancers..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " REBUILD COMPLETE - EasyVisa V12 - $CLUSTER_NAME" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

kubectl get pods -n $NAMESPACE
Write-Host ""
kubectl get svc -n $NAMESPACE

$API_URL = kubectl get svc visa-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$GRAF_URL = kubectl get svc grafana-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$PROM_URL = kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$AF_URL = kubectl get svc airflow-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null

Write-Host ""
Write-Host "PUBLIC SERVICE URLs:" -ForegroundColor Cyan
if ($API_URL)  { Write-Host " API:        http://$API_URL/docs" -ForegroundColor Green }
if ($GRAF_URL) { Write-Host " Grafana:    http://$GRAF_URL  (admin/easyvisa123)" -ForegroundColor Green }
if ($PROM_URL) { Write-Host " Prometheus: http://$PROM_URL" -ForegroundColor Green }
if ($AF_URL)   { Write-Host " Airflow:    http://${AF_URL}:8080  (admin/easyvisa123)" -ForegroundColor Green }
Write-Host ""
Write-Host "NEXT: Run .\deploy_evidently.ps1 to push Evidently image" -ForegroundColor Yellow