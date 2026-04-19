# =============================================================================
# rebuild.ps1 — EasyVisa-Prod Full Cluster Rebuild (V10)
#
# USAGE:
#   .\rebuild.ps1
#
# WHAT THIS DOES:
#   1. Creates EKS cluster (eksctl)
#   2. Installs EBS CSI driver
#   3. Configures kubeconfig
#   4. Creates namespace
#   5. *** Creates Kubernetes secrets from .env ***  ← fixes CreateContainerConfigError
#   6. Deploys all services (MLflow, visa-api, Prometheus, Grafana, Evidently)
#   7. Registers model in MLflow
#   8. Waits for pods + shows public URLs
#
# PREREQUISITES:
#   - .env file with AWS credentials in project root
#   - eksctl, kubectl, aws CLI on PATH
#   - CI/CD already pushed correct :latest images to ECR
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

# -----------------------------------------------------------------------------
# STEP 0: Load .env
# -----------------------------------------------------------------------------
Write-Host "`n[0/8] Loading .env credentials..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Error ".env file not found in project root. Aborting."
    exit 1
}
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.+)$') {
        $key   = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}
Write-Host "   AWS_ACCESS_KEY_ID loaded: $($env:AWS_ACCESS_KEY_ID.Substring(0,8))***" -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 1: Create EKS Cluster
# -----------------------------------------------------------------------------
Write-Host "`n[1/8] Creating EKS cluster '$CLUSTER_NAME'..." -ForegroundColor Yellow
eksctl create cluster `
    --name $CLUSTER_NAME `
    --region $REGION `
    --nodegroup-name easyvisa-nodes `
    --node-type t3.medium `
    --nodes 2 `
    --nodes-min 2 `
    --nodes-max 3 `
    --managed

Write-Host "   Cluster created." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 2: Install EBS CSI Driver
# -----------------------------------------------------------------------------
Write-Host "`n[2/8] Installing EBS CSI driver..." -ForegroundColor Yellow
$NODE_ROLE = aws iam list-roles `
    --query "Roles[?contains(RoleName, 'NodeInstanceRole')].RoleName" `
    --output text

if ($NODE_ROLE) {
    aws iam attach-role-policy `
        --role-name $NODE_ROLE `
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy 2>$null
    aws iam attach-role-policy `
        --role-name $NODE_ROLE `
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>$null
    Write-Host "   IAM policies attached to: $NODE_ROLE" -ForegroundColor Green
}

kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-ebs-csi-driver/master/deploy/kubernetes/overlays/stable/ecr/kustomization.yaml 2>$null
Write-Host "   EBS CSI driver installed." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 3: Update kubeconfig
# -----------------------------------------------------------------------------
Write-Host "`n[3/8] Updating kubeconfig..." -ForegroundColor Yellow
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION
Write-Host "   kubeconfig updated." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 4: Create Namespace
# -----------------------------------------------------------------------------
Write-Host "`n[4/8] Creating namespace '$NAMESPACE'..." -ForegroundColor Yellow
kubectl apply -f k8s/namespace.yaml
Write-Host "   Namespace ready." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 5: Create Kubernetes Secret (THE CRITICAL FIX)
# -----------------------------------------------------------------------------
Write-Host "`n[5/8] Creating Kubernetes secret 'easyvisa-secrets'..." -ForegroundColor Yellow


# Open rebuild.ps1 and find the secret creation block, replace with:
kubectl create secret generic easyvisa-secrets `
    --namespace=$NAMESPACE `
    --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" `
    --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" `
    --from-literal=aws-default-region="$REGION" `
    --from-literal=mlflow-tracking-uri="http://mlflow:5000" `
    --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production" `
    --dry-run=client -o yaml | kubectl apply -f -    

Write-Host "   Secret 'easyvisa-secrets' created/updated." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 6: Deploy All Services
# -----------------------------------------------------------------------------
Write-Host "`n[6/8] Deploying services to EKS..." -ForegroundColor Yellow

# Update ECR image references in manifests
$ECR_BASE = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
(Get-Content k8s/deployment.yaml) `
    -replace 'ACCOUNT_ID\.dkr\.ecr\.REGION\.amazonaws\.com', $ECR_BASE |
    Set-Content k8s/deployment.yaml

(Get-Content k8s/evidently-deployment.yaml) `
    -replace 'ACCOUNT_ID\.dkr\.ecr\.REGION\.amazonaws\.com', $ECR_BASE |
    Set-Content k8s/evidently-deployment.yaml

# Apply all manifests
kubectl apply -f k8s/mlflow-deployment.yaml    -n $NAMESPACE
kubectl apply -f k8s/deployment.yaml           -n $NAMESPACE
kubectl apply -f k8s/service.yaml              -n $NAMESPACE
kubectl apply -f k8s/prometheus-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s/grafana-deployment.yaml   -n $NAMESPACE
kubectl apply -f k8s/evidently-deployment.yaml -n $NAMESPACE

Write-Host "   All manifests applied." -ForegroundColor Green

# -----------------------------------------------------------------------------
# STEP 7: Register Model in MLflow
# -----------------------------------------------------------------------------
Write-Host "`n[7/8] Waiting 30s for MLflow to start, then registering model..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
python register_model.py
Write-Host "   Model registered." -ForegroundColor Green

# Restart visa-api to pick up the registered model
kubectl rollout restart deployment/visa-api-deployment -n $NAMESPACE

# -----------------------------------------------------------------------------
# STEP 8: Wait for Pods + Show Status
# -----------------------------------------------------------------------------
Write-Host "`n[8/8] Waiting 60s for pods to stabilize..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " REBUILD COMPLETE - $CLUSTER_NAME" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

kubectl get pods -n $NAMESPACE
Write-Host ""
kubectl get svc -n $NAMESPACE

Write-Host "`n[INFO] If evidently-service shows CreateContainerConfigError, run:" -ForegroundColor Yellow
Write-Host "  kubectl rollout restart deployment/evidently-service -n $NAMESPACE" -ForegroundColor White