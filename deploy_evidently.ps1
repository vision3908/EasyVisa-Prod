# deploy_evidently.ps1 — Run ONCE after rebuild.ps1
# Builds evidently image locally (has access to data/EasyVisa.csv)
# and pushes to ECR then deploys to EKS

$base = "167963468596.dkr.ecr.us-east-1.amazonaws.com/easyvisa-evidently"

Write-Host "Building evidently image locally..." -ForegroundColor Yellow
docker build --no-cache -f monitoring/Dockerfile.evidently -t "${base}:latest" .

Write-Host "Pushing to ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region us-east-1 | `
  docker login --username AWS --password-stdin `
  167963468596.dkr.ecr.us-east-1.amazonaws.com
docker push "${base}:latest"

Write-Host "Deploying to EKS..." -ForegroundColor Yellow
kubectl rollout restart deployment/evidently-service -n easyvisa

Start-Sleep -Seconds 45
kubectl logs -l app=evidently-service -n easyvisa --tail=20