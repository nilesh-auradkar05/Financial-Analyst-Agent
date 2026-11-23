import "server-only";
import { createCipheriv, createDecipheriv, createHmac, randomBytes, randomUUID, sign } from "node:crypto";
import { cookies } from "next/headers";

export const SESSION_COOKIE = "alpha_session";

export const SESSION_TTL_SECONDS = 24 * 3600;
const TOKEN_TTL_SECONDS = 900;
const IV_BYTES = 12;
const TAG_BYTES = 16;

export type GuestSession = { customer_id: string; ip_hash: string; exp: number };

const nowSeconds = (): number => Math.floor(Date.now() / 1000);

function aesKey(secret: string): Buffer {
  const key = Buffer.from(secret, "base64");
  if (key.length !== 32) throw new Error("session secret must be 32 bytes");
  return key;
}

/** Authenticated encryption (AES-256-GCM) of the session; the result is base64url, safe as a cookie value. */
export function sealSession(session: GuestSession, secret: string): string {
  const iv = randomBytes(IV_BYTES);
  const cipher = createCipheriv("aes-256-gcm", aesKey(secret), iv);
  const body = Buffer.concat([cipher.update(JSON.stringify(session), "utf8"), cipher.final()]);
  return Buffer.concat([iv, body, cipher.getAuthTag()]).toString("base64url");
}

/** The session behind a sealed value, or null when it is missing, forged, sealed under another key or expired. */
export function openSession(value: string | undefined, secret: string, now = nowSeconds()): GuestSession | null {
  if (!value) return null;
  try {
    const raw = Buffer.from(value, "base64url");
    if (raw.length <= IV_BYTES + TAG_BYTES) return null;
    const decipher = createDecipheriv("aes-256-gcm", aesKey(secret), raw.subarray(0, IV_BYTES));
    decipher.setAuthTag(raw.subarray(raw.length - TAG_BYTES));
    const plain = Buffer.concat([decipher.update(raw.subarray(IV_BYTES, raw.length - TAG_BYTES)), decipher.final()]);
    const s = JSON.parse(plain.toString("utf8")) as Partial<GuestSession>;
    if (typeof s.customer_id !== "string" || typeof s.ip_hash !== "string" || typeof s.exp !== "number" || s.exp <= now) return null;
    return { customer_id: s.customer_id, ip_hash: s.ip_hash, exp: s.exp };
  } catch {
    return null;
  }
}

/** A fresh guest identity; only a salted hash of the client IP is kept (the quota key), never the IP. */
export function newGuestSession(ip: string, salt: string, now = nowSeconds()): GuestSession {
  return {
    customer_id: `guest:${randomUUID()}`,
    ip_hash: createHmac("sha256", salt).update(ip).digest("hex"),
    exp: now + SESSION_TTL_SECONDS,
  };
}

/** The visitor's session from the request cookie, or null (also when SESSION_SECRET is unset). */
export async function currentSession(): Promise<GuestSession | null> {
  const secret = process.env.SESSION_SECRET;
  if (!secret) return null;
  return openSession((await cookies()).get(SESSION_COOKIE)?.value, secret);
}

type MintOptions = { privateKeyPem: string; issuer: string; audience: string; nowSeconds?: number };

/** Compact ES256 guest access token (15 minutes) for the API; it never leaves the server. */
export function mintGuestToken(session: GuestSession, opts: MintOptions): string {
  const iat = opts.nowSeconds ?? nowSeconds();
  const part = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
  const input = `${part({ alg: "ES256", typ: "JWT" })}.${part({
    iss: opts.issuer,
    aud: opts.audience,
    sub: session.customer_id,
    role: "guest",
    ip_hash: session.ip_hash,
    iat,
    exp: iat + TOKEN_TTL_SECONDS,
  })}`;
  const signature = sign("sha256", Buffer.from(input), { key: opts.privateKeyPem, dsaEncoding: "ieee-p1363" });
  return `${input}.${signature.toString("base64url")}`;
}
