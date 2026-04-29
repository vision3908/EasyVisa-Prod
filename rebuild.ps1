# =============================================================================
# rebuild.ps1 — EasyVisa-Prod Full Cluster Rebuild (V13)
#
# USAGE:  .\rebuild.ps1
#
# WHAT THIS DOES:
#   0. Loads .env credentials
#   1. Creates EKS cluster
#   2. Installs EBS CSI driver + OIDC
#   3. Updates kubeconfig
#   4. Creates namespace
#   5. Creates Kubernetes secret from .env (THE CRITICAL FIX)
#   6. Deploys MLflow first, waits for ready
#   7. Registers model inside cluster (no port-forward)
#   8. Deploys all other services
#   9. Shows status + public URLs
#
# PREREQUISITES:
#   - .env file with AWS credentials in project root
#   - eksctl, kubectl, aws CLI on PATH
#   - Docker images already pushed to ECR via CI/CD
# =============================================================================

$ErrorActionPreference = "Continue"
$CLUSTER_NAME = "easyvisa-cluster-v3"
$REGION       = "us-east-1"
$NAMESPACE    = "easyvisa"
$ACCOUNT_ID   = "167963468596"
$S3_MODEL_URI = "s3://easyvisa-mlflow-vision-2025/mlflow-artifacts/b6e88fbd91244f93b1bd5b2e36c2418a/artifacts/model"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " EasyVisa-Prod Rebuild V13" -ForegroundColor Cyan
Write-Host " Cluster : $CLUSTER_NAME  |  Region: $REGION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# STEP 0: Load .env
# -----------------------------------------------------------------------------
Write-Host "`n[0/9] Loading .env credentials..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Error ".env not found in project root. Aborting."
    exit 1
}
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
Write-Host "   AWS_ACCESS_KEY_ID: $($env:AWS_ACCESS_KEY_ID.Substring(0,8))***" -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 1: Create EKS Cluster
# -----------------------------------------------------------------------------
Write-Host "`n[1/9] Creating EKS cluster '$CLUSTER_NAME'..." -ForegroundColor Yellow
eksctl create cluster `
    --name $CLUSTER_NAME `
    --region $REGION `
    --nodegroup-name easyvisa-nodes `
    --node-type t3.medium `
    --nodes 3 `
    --nodes-min 2 `
    --nodes-max 4 `
    --managed
Write-Host "   Cluster created." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 2: Install EBS CSI Driver with OIDC
# -----------------------------------------------------------------------------
Write-Host "`n[2/9] Installing EBS CSI driver..." -ForegroundColor Yellow

eksctl utils associate-iam-oidc-provider `
    --cluster $CLUSTER_NAME --region $REGION --approve

eksctl create iamserviceaccount `
    --name ebs-csi-controller-sa `
    --namespace kube-system `
    --cluster $CLUSTER_NAME `
    --region $REGION `
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy `
    --approve `
    --override-existing-serviceaccounts

eksctl create addon `
    --name aws-ebs-csi-driver `
    --cluster $CLUSTER_NAME `
    --region $REGION `
    --force

Write-Host "   EBS CSI driver installed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 3: Update kubeconfig
# -----------------------------------------------------------------------------
Write-Host "`n[3/9] Updating kubeconfig..." -ForegroundColor Yellow
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
Write-Host "   kubeconfig updated." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 4: Create Namespace
# -----------------------------------------------------------------------------
Write-Host "`n[4/9] Creating namespace '$NAMESPACE'..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
Write-Host "   Namespace ready." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 5: Create Kubernetes Secret from .env (CRITICAL)
# -----------------------------------------------------------------------------
Write-Host "`n[5/9] Creating Kubernetes secret 'easyvisa-secrets'..." -ForegroundColor Yellow

kubectl create secret generic easyvisa-secrets `
    --namespace=$NAMESPACE `
    --from-literal=AWS_ACCESS_KEY_ID="$env:AWS_ACCESS_KEY_ID" `
    --from-literal=AWS_SECRET_ACCESS_KEY="$env:AWS_SECRET_ACCESS_KEY" `
    --from-literal=AWS_DEFAULT_REGION="$env:AWS_DEFAULT_REGION" `
    --from-literal=MLFLOW_TRACKING_URI="http://mlflow:5000" `
    --from-literal=mlflow-tracking-uri="http://mlflow:5000" `
    --from-literal=MLFLOW_MODEL_URI="models:/easyvisa_gbm/Production" `
    --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production" `
    --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" `
    --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" `
    --from-literal=aws-default-region="$env:AWS_DEFAULT_REGION" `
    --dry-run=client -o yaml | kubectl apply -f -

Write-Host "   Secret created." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 6: Deploy MLflow FIRST and wait
# -----------------------------------------------------------------------------
Write-Host "`n[6/9] Deploying MLflow..." -ForegroundColor Yellow
kubectl apply -f k8s/mlflow-deployment.yaml -n $NAMESPACE

Write-Host "   Waiting for MLflow to be ready (up to 3 min)..." -ForegroundColor Yellow
kubectl rollout status deployment/mlflow -n $NAMESPACE --timeout=180s
Write-Host "   MLflow ready." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 7: Register model inside cluster (no port-forward needed)
# -----------------------------------------------------------------------------
Write-Host "`n[7/9] Registering model in MLflow from inside cluster..." -ForegroundColor Yellow

# Write registration script
@'
import mlflow, sys
mlflow.set_tracking_uri('http://mlflow:5000')
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
    's3://easyvisa-mlflow-vision-2025/mlflow-artifacts/b6e88fbd91244f93b1bd5b2e36c2418a/artifacts/model',
    run_id
)
client.transition_model_version_stage('easyvisa_gbm', mv.version, 'Production')
print('Done - version:', mv.version)
'@ | Out-File -Encoding utf8 register_model.py

# Upload feature_names.pkl to S3 alongside model artifacts
aws s3 cp feature_names.pkl s3://easyvisa-mlflow-vision-2025/mlflow-artifacts/1/models/m-f516ecb85f7d405baca9efdac414171f/artifacts/feature_names.pkl
Write-Host "   feature_names.pkl uploaded to S3." -ForegroundColor Green

kubectl create configmap register-script `
    --from-file=register_model.py `
    -n $NAMESPACE `
    --dry-run=client -o yaml | kubectl apply -f -

kubectl run register-model --rm -it --restart=Never `
    --image=ghcr.io/mlflow/mlflow:v2.10.0 `
    --env="MLFLOW_TRACKING_URI=http://mlflow:5000" `
    --env="AWS_ACCESS_KEY_ID=$env:AWS_ACCESS_KEY_ID" `
    --env="AWS_SECRET_ACCESS_KEY=$env:AWS_SECRET_ACCESS_KEY" `
    --env="AWS_DEFAULT_REGION=us-east-1" `
    --overrides="{`"spec`":{`"volumes`":[{`"name`":`"script`",`"configMap`":{`"name`":`"register-script`"}}],`"containers`":[{`"name`":`"register-model`",`"image`":`"ghcr.io/mlflow/mlflow:v2.10.0`",`"command`":[`"python`",`"/scripts/register_model.py`"],`"volumeMounts`":[{`"name`":`"script`",`"mountPath`":`"/scripts`"}],`"env`":[{`"name`":`"MLFLOW_TRACKING_URI`",`"value`":`"http://mlflow:5000`"},{`"name`":`"AWS_ACCESS_KEY_ID`",`"value`":`"$env:AWS_ACCESS_KEY_ID`"},{`"name`":`"AWS_SECRET_ACCESS_KEY`",`"value`":`"$env:AWS_SECRET_ACCESS_KEY`"},{`"name`":`"AWS_DEFAULT_REGION`",`"value`":`"us-east-1`"}]}]}}" `
    -n $NAMESPACE

Write-Host "   Model registered." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 8: Deploy all other services
# -----------------------------------------------------------------------------
Write-Host "`n[8/9] Deploying all services..." -ForegroundColor Yellow

kubectl apply -f k8s/deployment.yaml           -n $NAMESPACE
kubectl apply -f k8s/service.yaml              -n $NAMESPACE
kubectl apply -f k8s/grafana-deployment.yaml   -n $NAMESPACE
kubectl apply -f k8s/prometheus-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/airflow-deployment.yaml   -n $NAMESPACE
kubectl apply -f k8s/evidently-deployment.yaml -n $NAMESPACE

# Open NodePort security group rules
Write-Host "   Opening NodePort security group rules..." -ForegroundColor Yellow
$NODE_SG = aws ec2 describe-instances `
    --region $REGION `
    --filters "Name=tag:eks:cluster-name,Values=$CLUSTER_NAME" `
    --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' `
    --output text

if ($NODE_SG -and $NODE_SG -ne "None") {
    aws ec2 authorize-security-group-ingress --group-id $NODE_SG --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $REGION 2>$null
    aws ec2 authorize-security-group-ingress --group-id $NODE_SG --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $REGION 2>$null
    aws ec2 authorize-security-group-ingress --group-id $NODE_SG --protocol tcp --port 30000 --port 32767 --cidr 0.0.0.0/0 --region $REGION 2>$null
    Write-Host "   Security group $NODE_SG updated." -ForegroundColor Green
}

Write-Host "   All services deployed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 9: Show status + URLs
# -----------------------------------------------------------------------------
Write-Host "`n[9/9] Cluster status..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

kubectl get pods -n $NAMESPACE
Write-Host ""
kubectl get svc -n $NAMESPACE

$API_URL     = kubectl get svc visa-api-service  -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$GRAFANA_URL = kubectl get svc grafana-service    -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$AIRFLOW_URL = kubectl get svc airflow-service    -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$PROM_URL    = kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " PUBLIC SERVICE URLs:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host " API:        http://$API_URL/docs" -ForegroundColor Green
Write-Host " Grafana:    http://$GRAFANA_URL (admin/easyvisa123)" -ForegroundColor Green
Write-Host " Airflow:    http://$($AIRFLOW_URL):8080 (admin/easyvisa123)" -ForegroundColor Green
Write-Host " Prometheus: http://$PROM_URL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host " REBUILD COMPLETE - EasyVisa V13" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green