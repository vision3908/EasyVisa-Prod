# -------------------------------------------------------
# EASYVISA CLUSTER REBUILD SCRIPT - FULLY AUTOMATED
# Usage: .\rebuild.ps1
# Change CLUSTER_NAME if previous cluster is stuck
# -------------------------------------------------------

$CLUSTER_NAME   = "easyvisa-cluster-v3"
$NODEGROUP_NAME = "easyvisa-nodes"
$REGION         = "us-east-1"
$NAMESPACE      = "easyvisa"
$ACCOUNT_ID     = "167963468596"
$ECR             = "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# 1. Load env vars from .env
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}

# 2. Create cluster
eksctl create cluster `
  --name $CLUSTER_NAME `
  --region $REGION `
  --nodegroup-name $NODEGROUP_NAME `
  --node-type t3.medium `
  --nodes 2

# 3. Update kubeconfig
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

# 4. Install EBS CSI driver
aws eks create-addon `
  --cluster-name $CLUSTER_NAME `
  --addon-name aws-ebs-csi-driver `
  --region $REGION

# 5. Get node role and attach EBS policy
$nodeRoleArn = (aws eks describe-nodegroup `
  --cluster-name $CLUSTER_NAME `
  --nodegroup-name $NODEGROUP_NAME `
  --region $REGION `
  --query "nodegroup.nodeRole" `
  --output text).Trim()
$nodeRole = $nodeRoleArn.Split("/")[-1]
Write-Host "Node role: [$nodeRole]"

aws iam attach-role-policy `
  --role-name $nodeRole `
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy

# 6. Wait for EBS CSI addon
Write-Host "Waiting for EBS CSI driver (90s)..."
Start-Sleep -Seconds 90

# 7. Create namespace
kubectl apply -f k8s\namespace.yaml

# 8. Create secret
kubectl create secret generic easyvisa-secrets -n $NAMESPACE `
  --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" `
  --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" `
  --from-literal=aws-default-region="$REGION" `
  --from-literal=mlflow-tracking-uri="http://mlflow:5000" `
  --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production"

# 9. Apply MLflow PVC and deployment
kubectl apply -f k8s\mlflow-pvc.yaml
kubectl apply -f k8s\mlflow-deployment.yaml

# 10. Wait for MLflow to be ready
Write-Host "Waiting for MLflow (120s)..."
Start-Sleep -Seconds 120

# 11. Apply remaining services
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl apply -f k8s\grafana-deployment.yaml
kubectl apply -f k8s\prometheus-deployment.yaml
kubectl apply -f k8s\evidently-deployment.yaml

# 12. Fix evidently ECR placeholder
kubectl set image deployment/evidently-service `
  evidently-service=$ECR/easyvisa-evidently:latest `
  -n $NAMESPACE

# 13. Add console access for Administrator
eksctl create iamidentitymapping `
  --cluster $CLUSTER_NAME `
  --region $REGION `
  --arn arn:aws:iam::${ACCOUNT_ID}:user/Administrator `
  --group system:masters `
  --username Administrator

# 14. Wait then register model
Write-Host "Waiting before model registration (30s)..."
Start-Sleep -Seconds 30

$job = Start-Job -ScriptBlock {
    kubectl port-forward svc/mlflow 5001:5000 -n easyvisa
}
Start-Sleep -Seconds 8

$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
python register_model.py

Stop-Job $job
Remove-Job $job

# 15. Restart visa-api to load registered model
kubectl rollout restart deployment/visa-api-deployment -n $NAMESPACE

# 16. Final status
Write-Host "REBUILD COMPLETE - $CLUSTER_NAME"
kubectl get pods -n $NAMESPACE
kubectl get svc -n $NAMESPACE
