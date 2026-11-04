# Heart Disease Prediction Using Machine Learning

## Task 78: Secure AWS Inference API

### Objective

Serve the versioned Task 47 model through FastAPI with bearer authentication, strict feature-schema/range validation, health checks, and confidence-aware output. Terraform provisions an HTTPS Application Load Balancer, private ECS Fargate tasks, scoped Secrets Manager access, CloudWatch logs, deployment rollback, and CPU/memory target-tracking autoscaling.

### Prerequisites

- AWS account and configured AWS CLI credentials with permission to manage ECR, ECS, IAM, ALB, CloudWatch, Secrets Manager, and Application Auto Scaling.
- Existing VPC with at least two public ALB subnets and two private task subnets with outbound access.
- ACM certificate and DNS name for the API.
- A Secrets Manager secret containing a randomly generated bearer token of at least 32 characters.
- The tracked model artifact at `Task_47_Versioned_Model_Export/artifacts/heart_logistic_v1.0.0.joblib`.

### Build and deploy

Create a local `terraform.tfvars` from `terraform.tfvars.example` with your region, VPC/subnets, certificate ARN, API domain, secret ARN, and the expected ECR URI. Do not commit this file or secret values. Configure an encrypted remote Terraform backend and state locking before production use.

```bash
cd Task_78_AWS_API_Deployment
terraform init
terraform apply -target=aws_ecr_repository.api
```

The ECR repository URL is a Terraform output. Authenticate Docker with ECR, then build from the repository root so the Dockerfile can copy `manual_ml.py` and the model artifact:

```bash
cd ..
docker build -f Task_78_AWS_API_Deployment/Dockerfile -t heart-model-api:1.0.0 .
```

Tag and push the image to the ECR URL output, update `image_uri` in your local tfvars to the immutable tag or digest, then run a full `terraform plan` and `terraform apply`. Create a DNS alias from `api_domain` to the ALB DNS output. Restrict AWS IAM permissions to the resources in use; the ECS task role is intentionally empty and only the execution role can retrieve the single API-token secret.

`GET /healthz` is unauthenticated for load-balancer checks. `POST /predict` requires `Authorization: Bearer ...`; the token is injected from Secrets Manager. Never put secrets in source control or load untrusted joblib artifacts.

### Verify

```bash
python -m pip install -r requirements-dev.txt
pytest -q test_api.py
```

This workspace cannot provision AWS resources because it has no account-specific network, certificate, DNS, secret, or credentials. Complete those values and deployment steps in your AWS account.

No notebook is included because API deployment, secret configuration, and Terraform provisioning are operational tasks; the runnable assets are the FastAPI service, container, tests, and Terraform files.