// ADR-0010 amendment 2026-10-11 §1 (guest role, server-side session, 15-minute ES256 token under the BFF's issuer);
// ADR-0010 §2 (the browser holds no bearer token); test-plan §20 "Guest access" and "Issuer allowlist".
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { AUDIENCE, ISSUER, newKeyPair, newSessionSecret, verifyEs256 } from "../test-support/fakes.ts";
import "../test-support/next-harness.ts";

const { SESSION_COOKIE, newGuestSession, openSession, sealSession, mintGuestToken } = await import("./session.server.ts");

const NOW = 1_800_000_000;
const SECRET = newSessionSecret();
const GUEST_UUID_V4 = /^guest:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

test("ADR-0010 amend. §1: session cookie is named alpha_session", () => {
  assert.equal(SESSION_COOKIE, "alpha_session");
});

test("test-plan §20 Guest access: a new guest session is guest:<uuid v4>, 24 h, with only a salted IP hash", () => {
  const ip = "203.0.113.7";
  const s = newGuestSession(ip, "salt-1", NOW);
  assert.match(s.customer_id, GUEST_UUID_V4);
  assert.equal(s.exp, NOW + 24 * 3600);
  assert.equal(s.ip_hash, createHmac("sha256", "salt-1").update(ip).digest("hex"));
  assert.equal(JSON.stringify(s).includes(ip), false, "raw IP must not be stored");

  assert.notEqual(newGuestSession(ip, "salt-1", NOW).customer_id, s.customer_id, "every visitor gets a fresh guest id");
  assert.equal(newGuestSession(ip, "salt-1", NOW).ip_hash, s.ip_hash, "same IP + salt hashes the same (quota key)");
  assert.notEqual(newGuestSession(ip, "salt-2", NOW).ip_hash, s.ip_hash, "the salt is part of the hash");
  assert.notEqual(newGuestSession("203.0.113.8", "salt-1", NOW).ip_hash, s.ip_hash);
});

test("ADR-0010 §2: a sealed session round-trips and reveals nothing in the cookie value", () => {
  const session = newGuestSession("198.51.100.4", "salt", NOW);
  const sealed = sealSession(session, SECRET);
  assert.deepEqual(openSession(sealed, SECRET, NOW + 60), session);
  assert.match(sealed, /^[^;,\s]+$/, "cookie-safe");
  assert.equal(sealed.includes(session.customer_id), false);
  assert.equal(sealed.includes(session.ip_hash), false);
  assert.equal(sealed.includes("guest"), false);
  assert.notEqual(sealSession(session, SECRET), sealed, "sealing uses a fresh nonce each time");
});

test("ADR-0010 §2: openSession returns null (never throws) for missing, tampered, truncated, foreign-key or expired values", () => {
  const session = newGuestSession("198.51.100.4", "salt", NOW);
  const sealed = sealSession(session, SECRET);
  const flip = (v: string, i: number) => v.slice(0, i) + (v[i] === "A" ? "B" : "A") + v.slice(i + 1);
  const cases: [string, string | undefined, string, number][] = [
    ["undefined", undefined, SECRET, NOW],
    ["empty", "", SECRET, NOW],
    ["garbage", "not-a-session", SECRET, NOW],
    ["flipped first char", flip(sealed, 0), SECRET, NOW],
    ["flipped middle char", flip(sealed, Math.floor(sealed.length / 2)), SECRET, NOW],
    ["flipped char near the end", flip(sealed, sealed.length - 5), SECRET, NOW],
    ["truncated", sealed.slice(0, sealed.length - 8), SECRET, NOW],
    ["half", sealed.slice(0, Math.floor(sealed.length / 2)), SECRET, NOW],
    ["different key", sealed, newSessionSecret(), NOW],
    ["secret is not 32 bytes", sealed, Buffer.from("short").toString("base64"), NOW],
    ["expired", sealed, SECRET, session.exp + 1],
  ];
  for (const [label, value, secret, now] of cases) {
    assert.equal(openSession(value, secret, now), null, label);
  }
  assert.deepEqual(openSession(sealed, SECRET, session.exp - 1), session, "still valid just before exp");
});

test("ADR-0010 amend. §1: openSession checks expiry against the real clock by default", () => {
  const nowS = Math.floor(Date.now() / 1000);
  const fresh = newGuestSession("198.51.100.4", "salt");
  assert.ok(Math.abs(fresh.exp - (nowS + 86400)) <= 5);
  assert.deepEqual(openSession(sealSession(fresh, SECRET), SECRET), fresh);
  const stale = { ...fresh, exp: nowS - 10 };
  assert.equal(openSession(sealSession(stale, SECRET), SECRET), null);
});

test("ADR-0010 amend. §1 + test-plan §20 Issuer allowlist: the minted token is a 15-minute ES256 guest JWT under the BFF issuer", () => {
  const { privateKeyPem, publicKey } = newKeyPair();
  const session = newGuestSession("203.0.113.7", "salt", NOW);
  const token = mintGuestToken(session, { privateKeyPem, issuer: ISSUER, audience: AUDIENCE, nowSeconds: NOW });

  const jwt = verifyEs256(token, publicKey);
  assert.ok(jwt, "signature verifies against the matching public key (ES256, IEEE P1363)");
  assert.equal(jwt.header.alg, "ES256");
  assert.equal(jwt.header.typ, "JWT");
  assert.equal(jwt.claims.iss, ISSUER);
  assert.equal(jwt.claims.aud, AUDIENCE);
  assert.equal(jwt.claims.sub, session.customer_id);
  assert.equal(jwt.claims.role, "guest");
  assert.equal(jwt.claims.ip_hash, session.ip_hash);
  assert.equal(jwt.claims.iat, NOW);
  assert.equal(jwt.claims.exp, NOW + 900);
});

test("test-plan §20 Issuer allowlist: a token does not verify under another key or after its claims are altered", () => {
  const signer = newKeyPair();
  const other = newKeyPair();
  const session = newGuestSession("203.0.113.7", "salt", NOW);
  const token = mintGuestToken(session, { privateKeyPem: signer.privateKeyPem, issuer: ISSUER, audience: AUDIENCE, nowSeconds: NOW });
  assert.equal(verifyEs256(token, other.publicKey), null);

  const [h, p, s] = token.split(".");
  const claims = JSON.parse(Buffer.from(p, "base64url").toString());
  const forged = Buffer.from(JSON.stringify({ ...claims, sub: "guest:someone-else", role: "admin" })).toString("base64url");
  assert.equal(verifyEs256(`${h}.${forged}.${s}`, signer.publicKey), null);
});

test("ADR-0010 amend. §1: the token's default issue time is the current clock", () => {
  const { privateKeyPem, publicKey } = newKeyPair();
  const before = Math.floor(Date.now() / 1000);
  const token = mintGuestToken(newGuestSession("203.0.113.7", "salt"), { privateKeyPem, issuer: ISSUER, audience: AUDIENCE });
  const claims = verifyEs256(token, publicKey)?.claims;
  assert.ok(claims);
  const iat = claims.iat as number;
  assert.ok(iat >= before && iat <= before + 5);
  assert.equal((claims.exp as number) - iat, 900);
});
