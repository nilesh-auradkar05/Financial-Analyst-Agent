import "server-only";

const SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify";
const TIMEOUT_MS = 5_000;
const ATTEMPTS = 2;

type Options = { secret: string | undefined; fetch?: typeof fetch };

/** True only when Cloudflare confirms the token. Fails closed; a network error or 5xx is retried once. */
export async function verifyTurnstile(token: string, remoteip: string | undefined, opts: Options): Promise<boolean> {
  if (!opts.secret || !token) return false;
  const body = JSON.stringify({ secret: opts.secret, response: token, remoteip });
  const doFetch = opts.fetch ?? globalThis.fetch;
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    let res: Response;
    try {
      res = await doFetch(SITEVERIFY, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
        cache: "no-store",
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch {
      continue;
    }
    if (res.status >= 500) continue;
    try {
      return ((await res.json()) as { success?: unknown }).success === true;
    } catch {
      return false;
    }
  }
  return false;
}
