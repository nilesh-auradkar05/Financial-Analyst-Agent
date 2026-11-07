# Financial Analyst Agent: audited Docker configurations

All six submitted files have corrected counterparts in this directory:

```text
Dockerfile                     # root/backend
.dockerignore                  # root/backend
frontend/Dockerfile            # was Dockerfile(1)
frontend/.dockerignore         # was .dockerignore(1)
docker-compose.yml             # main stack, same persistent volume names
docker-compose.qdrant.yml     # isolated test instance + separate volume and ports
```

Additional file: `.env.example` (required settings). `docker-compose.gpu.yml` was removed on 2026-11-07 (CM-6d): the model plane is cloud-only, so no container needs a GPU.

## Important findings

1. Updated 2026-11-07 (CM-6b, CM-6d; ADR-0019, ADR-0011): sentiment (Claude Sonnet 5.5) and rerank (Cohere Rerank v5 fast) run on Microsoft Foundry, so `torch`, `transformers`, `sentence-transformers`, `triton` and every NVIDIA/CUDA package are gone from the lockfile and the image. Before that, `site-packages` was about 6.3 GB, mostly NVIDIA libraries (3.2 GB), `torch` (1.1 GB) and `triton` (894 MB), for FinBERT. Qwen3 embeddings are still sent to host Ollama until CM-6e.
2. Original `uv sync --frozen --group api` implicitly included default dependency groups (usually `dev`), unless configured otherwise. The new Dockerfile uses `--no-default-groups --group api`.
3. Original API `COPY . .` added 373 MB. New ignore rules keep datasets, model files, runtime files, caches, credentials, monitoring and frontend out of the API build context. Check whether any app runtime assets were inadvertently excluded and add explicit `COPY` instructions or mounts if necessary.
4. BuildKit caching now uses `RUN --mount=type=cache` for uv/npm downloads. These caches live outside the image but **still consume local disk**. Use `docker builder prune`/BuildKit GC to manage them. The largest improvement may be build-cache cleanup, not final runtime image size.
5. No container reserves a GPU (the FinBERT GPU override was removed with FinBERT on 2026-11-07). Host Ollama, used for embeddings until CM-6e, has its own setup outside Compose.
6. Existing Grafana hardcoded `admin/admin`, public ports were broadly bound, and Qdrant used `latest`. New configurations bind local ports to loopback, require secrets for app/Grafana, and use version tags. For external access use an authenticated TLS reverse proxy.
7. Standalone Qdrant previously used the same fixed container name and **same named volume** as full Compose, which risks concurrent access/storage corruption. The corrected standalone stack uses its own volume and ports 16333/16334. It therefore does **not** automatically contain the existing 916 MB Qdrant dataset.
8. Previous Next.js Dockerfile omitted `public/` and did not set cache ownership; the new one copies `public/` even when absent originally and runs as `node`.

## Required project conditions that could not be verified

- Root has `pyproject.toml` and `uv.lock`, with dependency group named `api`; existing build uses these. A *locked* build now fails if `uv.lock` does not match `pyproject.toml`, intentionally. Resolve it with `uv lock` and commit the lockfile.
- Backend is importable as `app.main:app` and `/health` is reachable without an API key. Existing files used this entry point and health endpoint, but API source was not provided. Change healthcheck if `/health` requires authorization or has a different path.
- Frontend has `package-lock.json`, the `build` npm script and `next.config.*` with `output: 'standalone'`. The Dockerfile validates the generated `server.js` during build. The config file itself was not included in the uploads.
- Paths under `monitoring/` exist, and the API doesn't import files excluded by `.dockerignore`; monitoring config was not supplied.
- Prometheus currently receives a Compose secret `api_key` at `/run/secrets/api_key`, but whether `prometheus.yml` uses it could not be checked. Docker Compose's `environment: API_KEY` secret source may require `API_KEY` exported in the **shell** environment as well as present in `.env`. If Compose reports it unset, run `export API_KEY='...'` before `docker compose up`. Secrets should ideally be read from files by the app; runtime env keys are preserved to avoid breaking existing code.
- Non-root application user UID/GID 1000 has sufficient permissions on existing volumes. This was unchanged from the original image.
- Optional pass-through Compose environment variables are deliberately left unset when absent rather than set to empty strings; this avoids breaking typed settings such as integer LLM token budgets.
- Sentiment and rerank need `AZURE_OPENAI_ENDPOINT`, `AZURE_API_KEY`, `AZURE_FOUNDRY_SENTIMENT_MODEL` (deployment `sonnet-5-5-sentiment`) and `AZURE_FOUNDRY_RERANK_MODEL` (deployment `cohere-rerank-v5-fast`) in `.env`; without the sentiment deployment a run reports a sentiment gap.

## How to install safely

1. Save backups of all existing persistent volumes before changing Qdrant or runtime versions. **Never use `docker compose down -v` to perform this upgrade.** Existing main-stack named volumes are intentionally unchanged. Your Qdrant image may currently be an older release than pinned `v1.19.2`; compare with the installed version and review storage-format migration notes first. Set `QDRANT_VERSION` in `.env` to the version you have *validated*, and upgrade deliberately rather than assuming compatibility.
2. Copy `Dockerfile`, `.dockerignore`, and the `docker-compose*.yml` files into project root; copy files in `frontend/` into existing project `frontend/` directory.
3. Make a `.env` using `.env.example` and set real `API_KEY`, `SEC_USER_AGENT`, and `GRAFANA_ADMIN_PASSWORD` values. Do not commit `.env`. If your Grafana persistent database already exists, `GF_SECURITY_ADMIN_PASSWORD` may **not** rotate the existing password; change it in Grafana or via its CLI/administration flow.
4. Before deploying, render the final Compose config and review it **privately**, as it may expose secrets:

   ```bash
   docker compose config --quiet
   ```

5. Build and run:

   ```bash
   docker compose up -d --build
   docker compose ps
   docker compose logs --tail=100 api
   ```

6. Verify:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   docker system df -v
   docker buildx du
   ```

   The Qwen3 embedding model is hosted in Ollama outside this API container and is not verified by these commands.

## Separate Qdrant-only tests

```bash
docker compose -p financial-analyst-qdrant-test -f docker-compose.qdrant.yml up -d
VECTOR_BACKEND=qdrant QDRANT_URL=http://127.0.0.1:16333 uv run pytest tests/unit/test_qdrant_store.py -q
```

The Qdrant-only stack uses `127.0.0.1:16333` (REST) and `127.0.0.1:16334` (gRPC), unlike the full stack's ports 6333/6334. Its new volume is called `qdrant-financial-analyst-standalone-test`, so full stack and test instance cannot accidentally write the same underlying Qdrant storage.

## Expected disk reduction

- Development dependencies: a modest reduction if `dev` was included previously.
- Local model packages: torch, triton and the NVIDIA libraries (about 5.2 GB of `site-packages`) left the image on 2026-11-07.
- Build-cache pruning: previously about 19.1 GB reclaimable. BuildKit automatic garbage collection is recommended in Docker Desktop/daemon configuration and/or builder configuration.
- Repo `COPY`: should shrink if the removed context contained disposable files; identify the 373 MB source with `du` before promising an exact reduction.

```bash
# Remove unused build cache (images, running data volumes retained)
docker builder prune -a
# For distinct Buildx builders, inspect/prune them separately:
docker buildx ls
docker buildx du
# WARNING: do NOT add --volumes to any prune command without a backup.
```

## Security/operational follow-up for a real production deployment

- Docker Compose on a local workstation is not a substitute for a controlled server ingress. Keep exposed ports bound to loopback; put HTTPS and auth in front of services for remote access.
- Existing `grafana/grafana:10.2.0` and `prom/prometheus:v2.47.0` image versions were retained for compatibility, but are old. Plan tested upgrades and run vulnerability scans (`docker scout quickview` or Trivy) rather than silently upgrading databases.
- Compose `API_KEY` and other external provider credentials remain injected as environment variables for compatibility with your app. A true secret-file migration requires corresponding app-side `_FILE` or file-reading support. Never send secret values as build args or expose them in browser `NEXT_PUBLIC_*` variables.
- Pin images by digest for truly immutable CI/CD builds and add an image scan + SBOM step. The updated Dockerfiles pin uv and the major language runtime/OS release, but base images still receive patched updates.
- No model weights are stored locally any more. The old `financial-analyst-model-cache` volume is no longer mounted; it was not deleted — remove it with `docker volume rm financial-analyst-model-cache` only after confirming it holds nothing you need.
- Host Ollama is configured via `host.docker.internal`. On Linux, host Ollama must listen on an address reachable via the bridge gateway; binding *only* 127.0.0.1 on the host may not suffice. Restrict any wider listening with a firewall.
- Qdrant service healthcheck checks TCP availability (like the existing one); it does **not** guarantee collection/index readiness. API client retries are still important during restart.
