# syntax=docker/dockerfile:1.7
# Next.js standalone production image. Requires output: 'standalone' in next.config.

ARG NODE_IMAGE=node:22-bookworm-slim
FROM ${NODE_IMAGE} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

FROM ${NODE_IMAGE} AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# The project's public/ directory is currently absent/empty; make COPY reliable.
RUN mkdir -p public && npm run build \
    && test -f .next/standalone/server.js

FROM ${NODE_IMAGE} AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

# Traced standalone output contains the minimal required runtime node_modules.
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
COPY --from=builder --chown=node:node /app/public ./public

# Next.js needs writable .next for image/prerender cache under the non-root user.
RUN mkdir -p /app/.next/cache && chown -R node:node /app/.next
USER node
EXPOSE 3000
CMD ["node", "server.js"]
