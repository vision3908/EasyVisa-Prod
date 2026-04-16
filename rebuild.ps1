# EASYVISA CLUSTER REBUILD SCRIPT
# Prerequisite: .env file has valid AWS credentials

# 1. Load env vars
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}

# 2. Create cluster
eksctl create cluster `
  --name easyvisa-cluster-v2 `
  --region us-east-1 `
  --nodegroup-name easyvisa-nodes `
  --node-type t3.medium `
  --nodes 2

# 3. Install EBS CSI driver
aws eks create-addon --cluster-name easyvisa-cluster-v2 --addon-name aws-ebs-csi-driver --region us-east-1

# 4. Attach EBS policy to node role
$nodeRole = (aws eks describe-nodegroup --cluster-name easyvisa-cluster-v2 --nodegroup-name easyvisa-nodes --region us-east-1 --query "nodegroup.nodeRole" --output text).Trim().Split("/")[-1]
aws iam attach-role-policy --role-name $nodeRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy

# 5. Create namespace
kubectl apply -f k8s/namespace.yaml

# 6. Create secret
kubectl create secret generic easyvisa-secrets -n easyvisa `
  --from-literal=aws-access-key-id="$env:AWS_ACCESS_KEY_ID" `
  --from-literal=aws-secret-access-key="$env:AWS_SECRET_ACCESS_KEY" `
  --from-literal=aws-default-region="us-east-1" `
  --from-literal=mlflow-tracking-uri="http://mlflow:5000" `
  --from-literal=mlflow-model-uri="models:/easyvisa_gbm/Production"

# 7. Apply all manifests
kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
Start-Sleep -Seconds 90  # Wait for MLflow

# 8. Apply remaining services
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/prometheus-deploy.yaml
kubectl apply -f k8s/evidently-deployme.yaml

# 9. Wait for MLflow ready then register model
Start-Sleep -Seconds 30
$job = Start-Job -ScriptBlock { kubectl port-forward svc/mlflow 5001:5000 -n easyvisa }
Start-Sleep -Seconds 5
python register_model.py
Stop-Job $job; Remove-Job $job

# 10. Restart visa-api
kubectl rollout restart deployment/visa-api-deployment -n easyvisa

Write-Host "DONE - check pods:"
kubectl get pods -n easyvisa