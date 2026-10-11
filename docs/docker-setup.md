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

Additional files: `docker-compose.gpu.yml` (NVIDIA override), `.env.example` (required settings).

## Important findings

1. `site-packages` is approximately 6.3 GB, mostly NVIDIA libraries (3.2 GB), `torch` (1.1 GB), and `triton` (894 MB). These are not automatically wasted if FinBERT does CUDA inference in the API. Qwen3 embeddings are currently sent to host Ollama, so CUDA inside the API is not needed *for those HTTP requests*.
2. Original `uv sync --frozen --group api` implicitly included default dependency groups (usually `dev`), unless configured otherwise. The new Dockerfile uses `--no-default-groups --group api`; this **does not remove GPU dependencies** specified in the lockfile's project/API group.
3. Original API `COPY . .` added 373 MB. New ignore rules keep datasets, model files, runtime files, caches, credentials, monitoring and frontend out of the API build context. Check whether any app runtime assets were inadvertently excluded and add explicit `COPY` instructions or mounts if necessary.
4. BuildKit caching now uses `RUN --mount=type=cache` for uv/npm downloads. These caches live outside the image but **still consume local disk**. Use `docker builder prune`/BuildKit GC to manage them. The largest improvement may be build-cache cleanup, not final runtime image size.
5. Previously no GPU devices were assigned to API; new opt-in GPU override reserves one NVIDIA GPU for API FinBERT, keeping the default file runnable without an NVIDIA host. Host Ollama requires its *own* GPU setup and may compete for VRAM.
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
- NVIDIA GPU enablement requires a configured NVIDIA driver + NVIDIA Container Toolkit on the Docker host.

## How to install safely

1. Save backups of all existing persistent volumes before changing Qdrant or runtime versions. **Never use `docker compose down -v` to perform this upgrade.** Existing main-stack named volumes are intentionally unchanged. Your Qdrant image may currently be an older release than pinned `v1.19.2`; compare with the installed version and review storage-format migration notes first. Set `QDRANT_VERSION` in `.env` to the version you have *validated*, and upgrade deliberately rather than assuming compatibility.
2. Copy `Dockerfile`, `.dockerignore`, and the two `docker-compose*.yml` files into project root; copy files in `frontend/` into existing project `frontend/` directory. The optional `docker-compose.gpu.yml` also goes in the repository root.
3. Make a `.env` using `.env.example` and set real `API_KEY`, `SEC_USER_AGENT`, and `GRAFANA_ADMIN_PASSWORD` values. Do not commit `.env`. If your Grafana persistent database already exists, `GF_SECURITY_ADMIN_PASSWORD` may **not** rotate the existing password; change it in Grafana or via its CLI/administration flow.
4. Before deploying, render the final Compose config and review it **privately**, as it may expose secrets:

   ```bash
   docker compose config --quiet
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
   ```

5. Build and run (GPU FinBERT):

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   docker compose ps
   docker compose logs --tail=100 api
   ```

   Or without API GPU passthrough, use `docker compose up -d --build`.

6. Verify:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   docker compose exec api python -c "import torch; print('CUDA built:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU count:',torch.cuda.device_count())"
   docker system df -v
   docker buildx du
   ```

   `torch.cuda.is_available()` should report `True` in the API container for GPU FinBERT. The Qwen3 model is hosted in Ollama outside this API container and is not verified by that command.

## Separate Qdrant-only tests

```bash
docker compose -p financial-analyst-qdrant-test -f docker-compose.qdrant.yml up -d
VECTOR_BACKEND=qdrant QDRANT_URL=http://127.0.0.1:16333 uv run pytest tests/unit/test_qdrant_store.py -q
```

The Qdrant-only stack uses `127.0.0.1:16333` (REST) and `127.0.0.1:16334` (gRPC), unlike the full stack's ports 6333/6334. Its new volume is called `qdrant-financial-analyst-standalone-test`, so full stack and test instance cannot accidentally write the same underlying Qdrant storage.

## Expected disk reduction

- Development dependencies: a modest reduction if `dev` was included previously. Python GPU libraries can still legitimately consume >5 GB in your production image.
- Build-cache pruning: previously about 19.1 GB reclaimable, but rebuilding GPU dependencies creates large cache entries again. BuildKit automatic garbage collection is recommended in Docker Desktop/daemon configuration and/or builder configuration.
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
- Keep model weights in `financial-analyst-model-cache` instead of writing them into image layers. Volume storage still counts toward Docker disk use. Add an application-level retention or cache cleanup policy if model versions proliferate.
- Host Ollama is configured via `host.docker.internal`. On Linux, host Ollama must listen on an address reachable via the bridge gateway; binding *only* 127.0.0.1 on the host may not suffice. Restrict any wider listening with a firewall.
- Qdrant service healthcheck checks TCP availability (like the existing one); it does **not** guarantee collection/index readiness. API client retries are still important during restart.
