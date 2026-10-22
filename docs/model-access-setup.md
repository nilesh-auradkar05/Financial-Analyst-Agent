# Model Access Setup: Amazon Bedrock and Microsoft Foundry

Step-by-step setup for every cloud model the app calls, plus a one-call check per model. You run the checks and paste the output back into the task ledger (`tasks/todo.md`, MODEL-ACCESS-GUIDE). That output becomes the EMBED-ARMS access check.

Trace: EMBED-ARMS access check (ADR-0011 amendment 2026-10-11); ADR-0009 provider ladder (S7-AZURE-FALLBACK, S10-T03). Prices are as of 2026-10-16. Read the live price in each console before budgeting.

## 1. Which model does what

Proposed plan (user, 2026-10-16). It is recorded in ADR-0009 only after the amendment is accepted.

| Role | Primary | Fallback |
| --- | --- | --- |
| Memo generation (draft) | **Azure:** DeepSeek-V4-Flash, Global Standard | **Bedrock:** DeepSeek V3.2 |
| Judge / verifier | **Bedrock:** GPT-5.6 Luna, `global.openai.gpt-5.6-luna` | **Azure:** GPT-5.6-luna (short context), Global Standard |
| Embedding arms (EMBED-ARMS bake-off) | Bedrock: Titan Text Embeddings V2 and Cohere Embed v4, 1024 dims | Azure: `text-embedding-3-small`, 1024 dims |

The embedding rows are bake-off arms, not a primary/fallback pair. The model change above does not affect them, and the embedding checks can run now.

Prices per million tokens (input / output):

| Model | Host | Price | Source |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash | Foundry, Global Standard | $0.19 / $0.51 (cached input $0.028) | Microsoft Foundry launch post (about 100 days old; one guide says $0.20 / $0.80) |
| DeepSeek V3.2 | Bedrock | $0.62 / $1.85 | ADR-0009, 2026-10-08; re-check |
| GPT-5.6 Luna, short context (≤272K input) | Bedrock, Global CRIS | $0.20 / $1.20 (Geo or in-Region $0.22 / $1.32) | AWS model card |
| GPT-5.6-luna (short context) | Foundry, Global Standard | $0.20 / $1.20 | Microsoft Foundry GPT-5.6 post |


### What the plan costs

- **The fallback is a different model.** V4-Flash and V3.2 are different generations and sizes, so the Bedrock fallback needs its own quality check on the frozen eval set before it is trusted.
- **New memo baseline needed.** The memo results in `evaluation/results/` were made with `deepseek.v3.2` at temperature 0.3, so they describe the fallback. V4-Flash needs a fresh baseline, and it must not be compared with V3.2 results while anything else changes.
- **An Azure outage raises the bill.** Drafting falls back to V3.2 at about 3 to 3.6 times the per-token price. Set a budget alert on both clouds.


## 2. Amazon Bedrock setup

Region: `us-east-1`, matching the app's `AWS_REGION` default.

### 2.1 Bedrock API key

1. Sign in to the AWS console as a user who can manage IAM.
2. Open **Amazon Bedrock → API keys** and choose **Generate long-term API key**. Set an expiry; 30 days is fine for testing.
3. The console creates an IAM user for the key with `AmazonBedrockLimitedAccess` attached. Copy the key once; it is not shown again.
4. This is the value of `AWS_BEARER_TOKEN_BEDROCK`. The app already reads it, and so do boto3 and the AWS CLI.

### 2.2 Extra permission for GPT-5.6 Luna

Luna on `bedrock-runtime` also needs `bedrock:InvokeModel` on the account's default project. Add this inline policy to the key's IAM user, replacing `ACCOUNT_ID`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "bedrock:InvokeModel",
    "Resource": "arn:aws:bedrock:us-east-1:ACCOUNT_ID:project/default"
  }]
}
```

### 2.3 Model access

Bedrock enables serverless models on first call. Third-party models sold through AWS Marketplace (the GPT-5.6 family has a Marketplace product id) need the **first** call to come from an identity allowed `aws-marketplace:Subscribe`. If the Luna check fails with an access or subscription error, open the Luna model card in the console once and accept the offer, then rerun.

### 2.4 How the app calls Bedrock

The app calls Bedrock through the Converse API on `https://bedrock-runtime.us-east-1.amazonaws.com` (`ChatBedrockConverse`). Every Bedrock model in this plan works on that path:

- DeepSeek V3.2 by its bare id `deepseek.v3.2` (verified 2026-10-17; `us.deepseek.v3.2` does not exist).
- Luna only through a cross-region profile: `global.openai.gpt-5.6-luna` (cheapest) or `us.openai.gpt-5.6-luna`. The bare `openai.gpt-5.6-luna` is not accepted on `bedrock-runtime`.

**Luna on this account (2026-10-17):** both profiles return `AccessDeniedException: openai.gpt-5.6-luna is not available for this account`. The model is entitled and authorized, but the account's `bedrock-runtime` cross-region quotas for GPT-5.6 Luna are **0** tokens/min (geo and global). The `bedrock-mantle` quotas are not zero (20M input and 2M output tokens/min). So Luna is reachable only through `bedrock-mantle`, the OpenAI-compatible endpoint `https://bedrock-mantle.us-east-1.api.aws/openai/v1`, with the bare id `openai.gpt-5.6-luna` and the Bedrock API key. Check (e2) tests that route. To use `bedrock-runtime` instead, request a quota increase in **Service Quotas → Amazon Bedrock → "Global cross-region model inference tokens per minute for GPT-5.6 Luna"**. If the quota cannot be adjusted, the error's advice applies: contact AWS Sales.

### 2.5 Access checks (Bedrock)

Run these from the repo root on your machine. Each one prints the model id, the result shape and token use, never a key. Put the key into your shell without echoing it or saving it to history:

```bash
read -rs AWS_BEARER_TOKEN_BEDROCK && export AWS_BEARER_TOKEN_BEDROCK
export AWS_REGION=us-east-1
```

**a. Confirm the exact model ids first.** AWS docs show Cohere v4 as `cohere.embed-v4` while other sources show `cohere.embed-v4:0`, and it may need an inference profile. These commands need the AWS CLI with normal IAM credentials; the bearer key alone is not enough for `list-*` calls.

```bash
aws bedrock list-foundation-models --region us-east-1 \
  --query "modelSummaries[?contains(modelId,'embed') || contains(modelId,'deepseek') || contains(modelId,'gpt-5.6')].[modelId, join(', ', inferenceTypesSupported)]" \
  --output table
aws bedrock list-inference-profiles --region us-east-1 \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'embed') || contains(inferenceProfileId,'deepseek') || contains(inferenceProfileId,'gpt-5.6')].inferenceProfileId" \
  --output table
```

If you cannot run these, skip them. Checks (c) and (d) try each candidate id and print the error, which also tells us which id is valid.

**b. Titan Text Embeddings V2, 1024 dimensions**

```bash
uv run python - <<'PY'
import json, boto3
c = boto3.client("bedrock-runtime", region_name="us-east-1")
body = {"inputText": "Apple 10-K risk factors", "dimensions": 1024, "normalize": True}
r = c.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps(body))
out = json.loads(r["body"].read())
print("titan-v2 keys:", sorted(out), "| dim:", len(out["embedding"]), "| tokens:", out.get("inputTextTokenCount"))
PY
```

**c. Cohere Embed v4, 1024 dimensions, query and document input types**

```bash
uv run python - <<'PY'
import json, boto3
from botocore.exceptions import ClientError
c = boto3.client("bedrock-runtime", region_name="us-east-1")
for MODEL in ("cohere.embed-v4:0", "cohere.embed-v4", "us.cohere.embed-v4:0"):
    for input_type in ("search_document", "search_query"):
        body = {"texts": ["Apple 10-K risk factors"], "input_type": input_type,
                "embedding_types": ["float"], "output_dimension": 1024}
        try:
            out = json.loads(c.invoke_model(modelId=MODEL, body=json.dumps(body))["body"].read())
        except ClientError as e:
            print(MODEL, input_type, "ERROR:", e.response["Error"]["Code"], e.response["Error"]["Message"][:160])
            continue
        emb = out["embeddings"]
        vec = emb["float"][0] if isinstance(emb, dict) else emb[0]
        print(MODEL, input_type, "| keys:", sorted(out), "| response_type:", out.get("response_type"),
              "| embeddings is", type(emb).__name__, "| dim:", len(vec))
PY
```

**d. DeepSeek V3.2 through Converse (draft fallback)**

ADR-0009 says V3.2 is called through the US inference profile, while the code preset uses the bare id. This check tries both, at the baseline temperature of 0.3.

```bash
uv run python - <<'PY'
import boto3
from botocore.exceptions import ClientError
c = boto3.client("bedrock-runtime", region_name="us-east-1")
msg = [{"role": "user", "content": [{"text": "In one sentence: what is a 10-K?"}]}]
for MODEL in ("deepseek.v3.2", "us.deepseek.v3.2"):
    try:
        r = c.converse(modelId=MODEL, messages=msg, inferenceConfig={"maxTokens": 200, "temperature": 0.3})
    except ClientError as e:
        print(MODEL, "ERROR:", e.response["Error"]["Code"], e.response["Error"]["Message"][:200]); continue
    text = " ".join(b.get("text", "") for b in r["output"]["message"]["content"])
    print(MODEL, "| stop:", r["stopReason"], "| usage:", r["usage"], "|", text[:200])
PY
```

**e. GPT-5.6 Luna through Converse (primary judge)**

The second call checks whether Luna accepts `temperature=0`, which a judge wants. GPT-5 family models often reject non-default temperature. If it errors, the judge code must omit the parameter.

```bash
uv run python - <<'PY'
import boto3
from botocore.exceptions import ClientError
c = boto3.client("bedrock-runtime", region_name="us-east-1")
msg = [{"role": "user", "content": [{"text": "Reply with exactly: SUPPORTED or UNSUPPORTED. Claim: Apple files a 10-K annually."}]}]
for cfg in ({"maxTokens": 500}, {"maxTokens": 500, "temperature": 0.0}):
    try:
        r = c.converse(modelId="global.openai.gpt-5.6-luna", messages=msg, inferenceConfig=cfg)
    except ClientError as e:
        print("luna", cfg, "ERROR:", e.response["Error"]["Code"], e.response["Error"]["Message"][:200]); continue
    text = " ".join(b.get("text", "") for b in r["output"]["message"]["content"])
    print("luna", cfg, "| stop:", r["stopReason"], "| usage:", r["usage"], "|", text[:200])
PY
```

**e2. GPT-5.6 Luna through `bedrock-mantle` (route with non-zero quota)**

Needs the Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`), not CLI credentials. The key's IAM user needs `bedrock-mantle:CreateInference` for `openai.gpt-5.6-luna` and `bedrock-mantle:CallWithBearerToken`. It tries Chat Completions and the Responses API, each with and without `temperature=0`.

```bash
uv run python - <<'PY'
import os
from openai import OpenAI, APIError
c = OpenAI(base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
           api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"])
prompt = "Reply with exactly: SUPPORTED or UNSUPPORTED. Claim: Apple files a 10-K annually."
for kw in ({}, {"temperature": 0.0}):
    try:
        r = c.chat.completions.create(model="openai.gpt-5.6-luna", max_completion_tokens=500,
                                      messages=[{"role": "user", "content": prompt}], **kw)
        print("mantle chat", kw, "| model:", r.model, "| finish:", r.choices[0].finish_reason,
              "| usage:", r.usage.prompt_tokens, r.usage.completion_tokens, "|", (r.choices[0].message.content or "")[:120])
    except APIError as err:
        print("mantle chat", kw, "ERROR:", type(err).__name__, str(err)[:200])
    try:
        r = c.responses.create(model="openai.gpt-5.6-luna", input=prompt, max_output_tokens=500, **kw)
        print("mantle responses", kw, "| model:", r.model, "| status:", r.status,
              "| usage:", r.usage.input_tokens, r.usage.output_tokens, "|", (r.output_text or "")[:120])
    except APIError as err:
        print("mantle responses", kw, "ERROR:", type(err).__name__, str(err)[:200])
PY
```

### 2.6 Bedrock settings for the app

The app reads these names today. Put the values in your local env file yourself; the agent never reads or prints it.

| Variable | Value | Read by the app today? |
| --- | --- | --- |
| `AWS_REGION` | `us-east-1` | Yes |
| `AWS_BEARER_TOKEN_BEDROCK` | the API key from 2.1 | Yes |
| `AWS_BEDROCK_MODEL` | `deepseek.v3.2` (verified by check d) | Yes |

Until the Azure tier is built, the app keeps today's order: Bedrock first, so V3.2 drafts every memo. That is a safe interim state; the Azure-first order arrives with the ADR-0009 amendment and its code. Judge and embedding variables come with the judge-provider change and with EMBED-ARMS (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`). Do not add them yet.

## 3. Microsoft Foundry setup

### 3.1 Create the Foundry resource and project

1. Go to <https://ai.azure.com> and sign in with the subscription you want billed.
2. Choose **Create new → Foundry resource**, then create a project. A new resource group such as `rg-alpha-analyst` keeps teardown easy.
3. Pick a region that offers **Global Standard** for all three models: DeepSeek-V4-Flash, GPT-5.6-luna and `text-embedding-3-small`. East US 2 or Sweden Central are common choices. Check the region-availability page for "Foundry Models sold by Azure" before creating the resource, because the region cannot be changed later.
4. In the Azure portal, open **Cost Management → Budgets** and add a monthly budget with an email alert for the resource group. Do the same in AWS Billing → Budgets.

### 3.2 Find the endpoint and key

1. Open the resource in the Azure portal → **Keys and Endpoint**, or the project's **Overview** page in Foundry.
2. Copy **Key 1** and the endpoint. Depending on the resource, the endpoint looks like `https://<resource>.openai.azure.com/` or `https://<resource>.services.ai.azure.com/`. Use the host your portal shows.
3. The app calls the OpenAI v1 route: `<endpoint>/openai/v1/`. No `api-version` parameter is needed on that route.
4. Production uses keyless sign-in (ADR-0009: an AWS STS token trusted by an Entra federated credential). The key is for local testing only.

### 3.3 Deploy the three models

In Foundry, open **Models + endpoints → Deploy model → Deploy base model**, then for each row:

| Catalog model | Deployment type | Deployment name (you choose; suggested) | Rate limit to start |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash (Direct from Azure) | Global Standard | `ds-v4-flash-draft` | 50K tokens/min |
| GPT-5.6-luna (short context) | Global Standard | `luna-judge` | 50K tokens/min |
| text-embedding-3-small | Global Standard | `embed-3-small` | 100K tokens/min |

Write down the three names exactly as typed; they are case-sensitive. These are the "deployment names" the app needs.

### 3.4 Access checks (Foundry)

```bash
read -rs AZURE_API_KEY && export AZURE_API_KEY
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"   # host from 4.2, no trailing /openai/v1
export AZ_DRAFT=ds-v4-flash-draft AZ_JUDGE=luna-judge AZ_EMBED=embed-3-small
```

**f. All three Azure deployments in one script**

For DeepSeek-V4-Flash it also prints whether the reply carries `reasoning_content`. If it does, thinking is on by default and is billed as output tokens.

```bash
uv run python - <<'PY'
import os
from openai import OpenAI, APIError
c = OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
           api_key=os.environ["AZURE_API_KEY"])
e = c.embeddings.create(model=os.environ["AZ_EMBED"], input="Apple 10-K risk factors", dimensions=1024)
print("azure embed | model:", e.model, "| dim:", len(e.data[0].embedding), "| tokens:", e.usage.total_tokens)
for name, temps in (("AZ_DRAFT", (None, 0.3)), ("AZ_JUDGE", (None, 0.0))):
    for t in temps:
        kw = {} if t is None else {"temperature": t}
        try:
            r = c.chat.completions.create(model=os.environ[name], max_completion_tokens=500,
                messages=[{"role": "user", "content": "In one sentence: what is a 10-K?"}], **kw)
        except APIError as err:
            print(name, kw, "ERROR:", type(err).__name__, str(err)[:200]); continue
        m = r.choices[0].message
        print(name, kw, "| model:", r.model, "| finish:", r.choices[0].finish_reason,
              "| usage:", r.usage.prompt_tokens, r.usage.completion_tokens,
              "| reasoning_content:", bool(getattr(m, "reasoning_content", None)), "|", (m.content or "")[:200])
PY
```

### 3.5 Foundry settings for the app

`docker-compose.yml` already passes these names through to the api container. The app does not call Azure yet (S7-AZURE-FALLBACK, now S10-T03).

| Variable | Value | Status |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | endpoint host from 4.2 | Passed through, not used yet |
| `AZURE_API_KEY` | Key 1 from 4.2 (local testing only) | Passed through, not used yet |
| `AZURE_FOUNDRY_MODEL` | `ds-v4-flash-draft` (draft deployment name) | Reserved by ADR-0009 |
| Judge and embedding deployment names | `luna-judge`, `embed-3-small` | Variable names are set by the ADR-0009 amendment and EMBED-ARMS |

## 4. What to paste back

Paste the printed lines from checks (a)–(f). Remove anything that looks like a key or an account id. From that output we record:

1. **Model ids:** Cohere v4 (with or without `:0`, profile or not), DeepSeek V3.2 (bare or `us.`), the Luna profile id.
2. **Request and response shapes:** whether Cohere returns `embeddings` as a list or a dict keyed by type; whether Luna accepts `temperature=0`; whether DeepSeek-V4-Flash thinks by default.
3. **Returned dimensions:** 1024 for Titan, Cohere and `text-embedding-3-small`.
4. **Any errors**, verbatim (code and message).

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ValidationException: ... on-demand throughput isn't supported` | Model needs an inference profile | Use the `us.` or `global.` id from check (a) |
| `AccessDeniedException` on Luna | Missing `project/default` permission or Marketplace subscription | Steps 3.2 and 3.3 |
| `404` or `ResourceNotFound` from Azure | Wrong deployment name or endpoint host | Use the name exactly as typed in 4.3 and the host from 4.2 |
| `400 Unsupported parameter: 'temperature'` | Reasoning model accepts only the default temperature | Omit `temperature` for that model; record it |
| `400 ... max_tokens is not supported` on Azure | GPT-5 family wants `max_completion_tokens` | The check already uses it; the app must too |
| DeepSeek-V4-Flash reply is empty with `finish: length` | Thinking used the whole token budget | Record it; the app sets thinking off or raises the limit |

## References

- [GPT-5.6 Luna model card (Amazon Bedrock)](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html)
- [DeepSeek models on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-deepseek.html)
- [Cohere Embed v4 request parameters (Amazon Bedrock)](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html)
- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)
- [DeepSeek V4 Flash and V4 Pro in Microsoft Foundry](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-deepseek-v4-flash-and-v4-pro-in-microsoft-foundry/4515174)
- [DeepSeek-V4.1-Flash is coming to Microsoft Foundry](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/deepseek-v4-1-flash-is-coming-to-microsoft-foundry/4556431)
- [GPT-5.6 in Microsoft Foundry: pricing](https://azure.microsoft.com/en-us/blog/gpt-5-6-now-available-in-microsoft-foundry/)
- [Foundry Models sold by Azure](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)
- [Foundry endpoints and the OpenAI v1 route](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/endpoints)
