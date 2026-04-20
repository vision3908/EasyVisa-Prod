# =============================================================================
# rebuild.ps1 - EasyVisa-Prod Full Cluster Rebuild (V10)
# =============================================================================

$ErrorActionPreference = "Stop"
$CLUSTER_NAME  = "easyvisa-cluster-v3"
$REGION        = "us-east-1"
$NAMESPACE     = "easyvisa"
$ACCOUNT_ID    = "167963468596"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " EasyVisa-Prod Rebuild Script V10" -ForegroundColor Cyan
Write-Host " Cluster: $CLUSTER_NAME | Region: $REGION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# STEP 0: Load .env
Write-Host "`n[0/8] Loading .env credentials..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) { Write-Error ".env not found. Aborting."; exit 1 }
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
    }
}
Write-Host "   AWS_ACCESS_KEY_ID: $($env:AWS_ACCESS_KEY_ID.Substring(0,8))***" -ForegroundColor Green

# STEP 1: Create EKS Cluster
Write-Host "`n[1/8] Creating EKS cluster..." -ForegroundColor Yellow
eksctl create cluster `
    --name $CLUSTER_NAME `
    --region $REGION `
    --nodegroup-name easyvisa-nodes `
    --node-type t3.medium `
    --nodes 2 --nodes-min 2 --nodes-max 3 `
    --managed
Write-Host "   Cluster created." -ForegroundColor Green

# STEP 2: Install EBS CSI Driver
Write-Host "`n[2/8] Installing EBS CSI driver..." -ForegroundColor Yellow
$NODE_ROLE = (aws iam list-roles --query "Roles[?contains(RoleName, 'NodeInstanceRole')].RoleName" --output text).Trim()
if ($NODE_ROLE -ne "") {
    Write-Host "   Attaching IAM policies to: $NODE_ROLE" -ForegroundColor Green
    aws iam attach-role-policy --role-name $NODE_ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy 2>$null
    aws iam attach-role-policy --role-name $NODE_ROLE --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>$null
}
eksctl create addon --name aws-ebs-csi-driver --cluster $CLUSTER_NAME --region $REGION --force
Write-Host "   EBS CSI driver installed." -ForegroundColor Green

# STEP 3: Update kubeconfig
Write-Host "`n[3/8] Updating kubeconfig..." -ForegroundColor Yellow
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
Write-Host "   kubeconfig updated." -ForegroundColor Green

# STEP 4: Create Namespace
Write-Host "`n[4/8] Creating namespace..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
Write-Host "   Namespace ready." -ForegroundColor Green

# STEP 5: Create Kubernetes Secret (kebab-case keys)
Write-Host "`n[5/8] Creating Kubernetes secret..." -ForegroundColor Yellow
kubectl create secret generic easyvisa-secrets `
    --namespace=$NAMESPACE `
    --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" `
    --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" `
    --from-literal=aws-default-region="$REGION" `
    --from-literal=mlflow-tracking-uri="http://mlflow:5000" `
    --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production" `
    --dry-run=client -o yaml | kubectl apply -f -
Write-Host "   Secret created." -ForegroundColor Green

# STEP 6: Deploy All Services
Write-Host "`n[6/8] Deploying services..." -ForegroundColor Yellow
$ECR_BASE = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
(Get-Content k8s/deployment.yaml) -replace 'ACCOUNT_ID\.dkr\.ecr\.REGION\.amazonaws\.com', $ECR_BASE | Set-Content k8s/deployment.yaml
(Get-Content k8s/evidently-deployment.yaml) -replace 'ACCOUNT_ID\.dkr\.ecr\.REGION\.amazonaws\.com', $ECR_BASE | Set-Content k8s/evidently-deployment.yaml

kubectl apply -f k8s/mlflow-deployment.yaml     -n $NAMESPACE
kubectl apply -f k8s/deployment.yaml            -n $NAMESPACE
kubectl apply -f k8s/service.yaml               -n $NAMESPACE
kubectl apply -f k8s/prometheus-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/grafana-deployment.yaml    -n $NAMESPACE
kubectl apply -f k8s/evidently-deployment.yaml  -n $NAMESPACE
kubectl apply -f k8s/hpa.yaml -n easyvisa       -n $NAMESPACE
Write-Host "   All manifests applied." -ForegroundColor Green

# STEP 7: Register Model in MLflow
Write-Host "`n[7/8] Waiting 90s for MLflow to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 90
$pfJob = Start-Job -ScriptBlock { kubectl port-forward svc/mlflow -n easyvisa 5000:5000 }
Start-Sleep -Seconds 10
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python register_model.py
Stop-Job $pfJob; Remove-Job $pfJob
Write-Host "   Model registered." -ForegroundColor Green
kubectl rollout restart deployment/visa-api-deployment -n $NAMESPACE

# STEP 8: Final Status
Write-Host "`n[8/8] Waiting 90s for pods to stabilize..." -ForegroundColor Yellow
Start-Sleep -Seconds 90

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " REBUILD COMPLETE - $CLUSTER_NAME" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

kubectl get pods -n $NAMESPACE
Write-Host ""
kubectl get svc -n $NAMESPACE

Write-Host "`n=== PUBLIC URLS ===" -ForegroundColor Cyan
$grafana = kubectl get svc grafana-service    -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$prom    = kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
$api     = kubectl get svc visa-api-service   -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>$null
if ($grafana) { Write-Host "Grafana   : http://$grafana"    -ForegroundColor Green }
if ($prom)    { Write-Host "Prometheus: http://$prom"       -ForegroundColor Green }
if ($api)     { Write-Host "API Docs  : http://$api/docs"   -ForegroundColor Green }
if ($api)     { Write-Host "API Health: http://$api/health" -ForegroundColor Green }

Write-Host "`n=== NEXT STEP ===" -ForegroundColor Yellow
Write-Host "Run: .\deploy_evidently.ps1" -ForegroundColor White