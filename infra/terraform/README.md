# infra/terraform

One stack per directory, applied from a clean checkout. State is local and gitignored
(one owner; move to an S3 backend when a second operator or CI applies). Every `apply`
that creates billable resources needs the owner's approval first (ADR-0016).

| Stack | Creates | Cost |
| --- | --- | --- |
| `cost/` (CM-1) | AWS Budgets monthly $100 budget (alerts at $10, $25, $100 actual and $100 forecast), the kill-switch IAM policy, the Budgets execution role, the $100 action | $0 |
| `search/` (CM-3) | OpenSearch Serverless NEXTGEN collection group `alpha-analyst-search` (0–2 indexing and 0–2 search OCU, scales to zero after about 10 idle minutes), `VECTORSEARCH` collection `alpha-analyst-vectors`, encryption (AWS-owned key), network (public, IAM auth only) and data access policies for one IAM user | $0.24 per OCU-hour while awake (about $0.48–0.96 per awake hour), $0.02 per GB-month; $0 idle compute |

## Cost stack

```bash
cd infra/terraform/cost
cat > terraform.tfvars <<'EOF'   # gitignored
alert_email       = "<owner email>"
kill_switch_users = ["<IAM users that call Bedrock and OpenSearch, incl. the API key user>"]
kill_switch_roles = ["<IAM roles assumed for those calls>"]
EOF
export AWS_PROFILE=<deployer profile>   # needs the CM-1 deployer policy
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

At $100 of actual monthly spend, AWS Budgets attaches `alpha-analyst-budget-kill-switch`
to the listed users and roles. It denies Bedrock invocations (incl. Rerank and `bedrock-mantle`)
and OpenSearch Serverless data access and creation; deletes stay allowed, so teardown still works. To resume after review, reset
the action in the AWS Budgets console (or detach the policy from the user or role).

Not covered: Azure Foundry spend (set an Azure budget in the Azure portal) and any
principal not listed in `kill_switch_users` or `kill_switch_roles` (including the deployer).

## Search stack

```bash
cd infra/terraform/search
echo 'data_access_user = "<IAM user that signs search requests>"' > terraform.tfvars   # gitignored
export AWS_PROFILE=<deployer profile>   # needs aoss collection-group, collection and policy create/delete
terraform init
terraform plan -out=tfplan
terraform apply tfplan
terraform output collection_endpoint   # → OPENSEARCH_URL, with OPENSEARCH_AUTH=sigv4
```

The data access user also needs an identity policy allowing `aoss:APIAccessAll` on the collection.

## Teardown

```bash
terraform -chdir=infra/terraform/search destroy
terraform -chdir=infra/terraform/cost destroy
```

Destroy the cost stack last: it is the guard for every other stack.
