import boto3
import os

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
)
s3.download_file(
    "easyvisa-mlflow-vision-2025",
    "data/EasyVisa.csv",
    "/app/data/EasyVisa.csv",
)
print("Downloaded EasyVisa.csv from S3 successfully")
