# AWS Non-Production Setup and Test Guide

This guide creates a disposable AWS EC2 environment for testing the Alpha Financial Analyst system. It is for non-production validation only.

This guide does not define production deployment architecture. It does not introduce Terraform/IaC, managed cloud rollout, public service hardening, autoscaling, production observability, or a scope change. Production cloud deployment remains out of scope unless SPEC, sprint planning, and any required ADR are updated first.

## AWS references

Use the current AWS documentation when account, operating system, networking, or CLI details differ from this guide:

- AWS CLI install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
- AWS CLI configuration files and profiles: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
- EC2 getting started: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html
- EC2 security groups: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html
- SSH to Linux EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html
- IAM roles for EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html
- Amazon Linux 2023: https://docs.aws.amazon.com/linux/al2023/ug/what-is-amazon-linux.html

## Test-environment rules

- Use a short-lived EC2 instance.
- Restrict inbound SSH to your current public IP.
- Do not expose the API, Qdrant, Ollama, Grafana, or Prometheus publicly.
- Use SSH port forwarding for API smoke tests.
- Do not place secrets in user data, shell history, committed files, screenshots, or task logs.
- Stop or terminate the instance when testing is complete.

## Prerequisites

On your local machine:

- AWS account access with permission to create EC2 instances, key pairs, and security groups.
- AWS CLI v2 installed and configured.
- A default VPC, or a known subnet ID for the target VPC.
- Git access to this repository.
- An SSH client.

Recommended instance sizes:

- `t3.large` or `t3.xlarge` for dependency install, static checks, and API smoke tests without local LLM load.
- A GPU instance, such as the appropriate current `g5` family option in your region, for serious live Ollama testing.

Size and cost depend on region, model choice, and runtime. Verify pricing before launch.

## Step 1 - Configure AWS CLI locally

Configure credentials and a default region:

```bash
aws configure
aws sts get-caller-identity
```

If you use IAM Identity Center or named profiles, configure the profile according to your organization policy and add `--profile <profile-name>` to the AWS commands in this guide.

Set local shell variables:

```bash
export AWS_REGION=us-east-1
export KEY_NAME=alpha-analyst-test
export SG_NAME=alpha-analyst-test-sg
export INSTANCE_NAME=alpha-analyst-test
export INSTANCE_TYPE=t3.xlarge
```

## Step 2 - Create a key pair and security group

Create a temporary SSH key pair:

```bash
aws ec2 create-key-pair \
  --region "$AWS_REGION" \
  --key-name "$KEY_NAME" \
  --query 'KeyMaterial' \
  --output text > "$KEY_NAME.pem"

chmod 400 "$KEY_NAME.pem"
```

Find your current public IP and default VPC:

```bash
export MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
export VPC_ID="$(aws ec2 describe-vpcs \
  --region "$AWS_REGION" \
  --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' \
  --output text)"
```

Create the security group:

```bash
export SG_ID="$(aws ec2 create-security-group \
  --region "$AWS_REGION" \
  --group-name "$SG_NAME" \
  --description 'Alpha Analyst non-production test access' \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' \
  --output text)"
```

Allow SSH only from your current IP:

```bash
aws ec2 authorize-security-group-ingress \
  --region "$AWS_REGION" \
  --group-id "$SG_ID" \
  --protocol tcp \
  --port 22 \
  --cidr "$MY_IP"
```

Do not add public inbound rules for ports `8000`, `6333`, `6334`, `11434`, `3000`, or `9090`.

## Step 3 - Launch a test EC2 instance

Resolve the current Amazon Linux 2023 AMI:

```bash
export AMI_ID="$(aws ssm get-parameter \
  --region "$AWS_REGION" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameter.Value' \
  --output text)"
```

Launch the instance:

```bash
export INSTANCE_ID="$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
  --query 'Instances[0].InstanceId' \
  --output text)"
```

If your account has no default subnet, add `--subnet-id <subnet-id>` to the launch command.

Wait for the instance and capture the public DNS name:

```bash
aws ec2 wait instance-running \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID"

export PUBLIC_DNS="$(aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' \
  --output text)"

echo "$PUBLIC_DNS"
```

## Step 4 - Connect over SSH

Connect to the instance:

```bash
ssh -i "$KEY_NAME.pem" ec2-user@"$PUBLIC_DNS"
```

All remaining setup commands are run on the EC2 instance unless noted otherwise.

## Step 5 - Install system tools on EC2

Update packages and install tools:

```bash
sudo dnf update -y
sudo dnf install -y git curl make docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
```

Confirm versions:

```bash
git --version
docker --version
uv --version
```

## Step 6 - Clone and install the project

Clone the repository:

```bash
git clone <repo-url>
cd Financial-Analyst-system
```

Synchronize dependencies:

```bash
uv sync --python 3.12
```

If the EC2 instance is small, dependency installation may be slow. Upgrade the instance type or add swap only as a temporary test-environment workaround.

## Step 7 - Configure environment on EC2

Create `.env` without secrets you do not need:

```bash
cat > .env <<'EOF'
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=qwen3-embedding:4b
OLLAMA_TIMEOUT=600
OLLAMA_TEMPERATURE=0.7

VECTOR_BACKEND=qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME=sec_filings

SEC_USER_AGENT="Your Name your-email@example.com"
TAVILY_API_KEY=

LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=financial-analyst-system
EOF
```

Replace the SEC user agent with your own contact string. Add a Tavily API key only for news-enabled live analysis.

## Step 8 - Start Qdrant bound to localhost

Run Qdrant without exposing it on the public network interface:

```bash
docker run -d \
  --name alpha-analyst-qdrant \
  -p 127.0.0.1:6333:6333 \
  -p 127.0.0.1:6334:6334 \
  -v alpha-analyst-qdrant:/qdrant/storage \
  qdrant/qdrant:latest

docker ps --filter name=alpha-analyst-qdrant
```

Check logs if startup fails:

```bash
docker logs alpha-analyst-qdrant
```

## Step 9 - Run verification on EC2

Run static and unit checks:

```bash
uv run ruff check .
uv run mypy app evaluation
uv run pytest -q
```

Run document governance checks:

```bash
python scripts/ci/check_no_scope_residue.py
python scripts/ci/check_sprint_map.py
python scripts/ci/check_doc_sync.py
python scripts/ci/check_test_hygiene.py
```

Run ingestion and retrieval checks:

```bash
uv run pytest tests/ingestion -q
VECTOR_BACKEND=qdrant QDRANT_URL=http://127.0.0.1:6333 uv run pytest tests/unit/test_qdrant_store.py -q
uv run python evaluation/validate_retrieval_fixture.py evaluation/fixtures/retrieval_shared_benchmark_v1.json
```

Record pass/fail results in `tasks/todo.md` if this EC2 run is part of an active project task.

## Step 10 - Run API smoke test through SSH tunnel

On EC2, start the API bound to localhost:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On your local machine, open a tunnel:

```bash
ssh -i "$KEY_NAME.pem" -L 8000:127.0.0.1:8000 ec2-user@"$PUBLIC_DNS"
```

In another local terminal, smoke test the API:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

`/health` can be `degraded` until Ollama and optional LangSmith are configured. The API should still return a structured response.

## Step 11 - Optional live analysis on EC2

Install and run Ollama only when live LLM and embedding tests are required. CPU-only instances can be very slow for the configured models.

After Ollama is running and models are pulled:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}'

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","include_filing_analysis":true,"include_news_sentiment":false,"max_news_articles":1}'
```

Enable news sentiment only after setting `TAVILY_API_KEY`.

## Step 12 - Teardown

On EC2, stop local containers:

```bash
docker stop alpha-analyst-qdrant
docker rm alpha-analyst-qdrant
docker volume rm alpha-analyst-qdrant
```

On your local machine, terminate the instance:

```bash
aws ec2 terminate-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID"

aws ec2 wait instance-terminated \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID"
```

Delete the security group:

```bash
aws ec2 delete-security-group \
  --region "$AWS_REGION" \
  --group-id "$SG_ID"
```

Delete the AWS key pair when you no longer need it:

```bash
aws ec2 delete-key-pair \
  --region "$AWS_REGION" \
  --key-name "$KEY_NAME"
```

Remove the local `.pem` file only after confirming there are no running instances that still require it.

## Troubleshooting

- SSH fails: confirm the instance is running, the public DNS is populated, the key permissions are `400`, and the security group allows port `22` from your current IP.
- Public IP changed: update `MY_IP` and replace the security group ingress rule.
- API is unreachable locally: confirm the SSH tunnel is open and the API is bound to `127.0.0.1:8000` on EC2.
- Qdrant is unreachable: confirm the container is running and `QDRANT_URL=http://127.0.0.1:6333`.
- Dependency install is slow or fails from memory pressure: use a larger test instance.
- Live analysis is slow: use a GPU instance or run only static and API smoke checks.
- News search fails: set `TAVILY_API_KEY` or disable news sentiment.
- SEC ingestion fails: check network access and `SEC_USER_AGENT`.
