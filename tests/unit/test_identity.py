"""S7.2 slice (a): API identity (ADR-0010 + 2026-10-11 amendment).

Trace: docs/test-plan.md §20 rows "Issuer allowlist", "Customer isolation", "Guest access" (API
side); §12 S7.2 rows "API auth", "Rate limit", "Idempotency" where the identity model changes them.
Everything is exercised through HTTP routes; tokens are minted here with throwaway EC P-256 keys.

S7.2 slice (c) changes: guest tokens now carry an `ip_hash` claim (a fresh one per token, so no
two tokens share an IP quota) and the `client` fixture installs fresh guest quota state, the way
it already installs a fresh `submission_limiter` (test-plan §20 Guest quotas).

Fixture split: `client` needs no new symbol (development issuer only); `allowlisted` imports
`JwtIssuer` from `app.config` and installs the allowlist, so every token test goes red on the
missing public symbol until it exists.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as api_main
from app.config import Settings
from app.services.run_store import FileBackedRunStore
from tests.test_api_integration import (
    StubVectorStore,
    fake_check_langsmith_connection,
    fake_ingest_10k_for_ticker,
    fake_run_agent,
)

API_KEY = "test-secret"
ISSUER_A = "https://site.example/guest"
AUDIENCE_A = "financial-analyst-api"
ISSUER_B = "https://cognito.example/pool"
AUDIENCE_B = "admin-console"
NOT_FOUND = "00000000-0000-4000-8000-000000000000"
MISSING = object()


@dataclass(frozen=True)
class KeyPair:
    private: ec.EllipticCurvePrivateKey
    public_pem: str


def _key_pair() -> KeyPair:
    private = ec.generate_private_key(ec.SECP256R1())
    pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return KeyPair(private, pem.decode())


KEY_A = _key_pair()
KEY_B = _key_pair()
KEY_UNLISTED = _key_pair()


def _claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER_A,
        "aud": AUDIENCE_A,
        "sub": "guest:" + str(uuid.uuid4()),
        "role": "guest",
        "ip_hash": uuid.uuid4().hex + uuid.uuid4().hex,
        "iat": now,
        "exp": now + 900,
        "jti": str(uuid.uuid4()),
    }
    claims.update(overrides)
    return {name: value for name, value in claims.items() if value is not MISSING}


def mint(key: KeyPair = KEY_A, **overrides: Any) -> str:
    return jwt.encode(_claims(**overrides), key.private, algorithm="ES256")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge(header: dict[str, Any], claims: dict[str, Any], hmac_secret: bytes | None) -> str:
    """Build a token by hand: an unsigned one, or an HMAC one (PyJWT refuses a PEM as secret)."""
    signing_input = _b64(json.dumps(header).encode()) + "." + _b64(json.dumps(claims).encode())
    if hmac_secret is None:
        return signing_input + "."
    signature = hmac.new(hmac_secret, signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64(signature)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


ADMIN_KEY_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding_arm_configured):
    monkeypatch.setattr(api_main, "run_store", FileBackedRunStore(tmp_path / "run_store.json"))
    monkeypatch.setattr(api_main, "get_vector_store", lambda: StubVectorStore())
    monkeypatch.setattr(api_main, "check_langsmith_connection", fake_check_langsmith_connection)
    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", fake_ingest_10k_for_ticker)
    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    monkeypatch.setattr(api_main.settings, "api_key", API_KEY)
    monkeypatch.setattr(api_main, "submission_limiter", api_main.SubmissionRateLimiter())
    monkeypatch.setattr(api_main, "guest_quotas", api_main.GuestQuotas())
    with TestClient(api_main.app) as test_client:
        yield test_client


def _install_allowlist(monkeypatch: pytest.MonkeyPatch, *issuers: tuple[str, str, KeyPair]) -> None:
    from app.config import JwtIssuer

    monkeypatch.setattr(
        api_main.settings,
        "jwt_issuers",
        [
            JwtIssuer(issuer=issuer, audience=audience, public_key_pem=key.public_pem)
            for issuer, audience, key in issuers
        ],
    )


@pytest.fixture()
def allowlisted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _install_allowlist(monkeypatch, (ISSUER_A, AUDIENCE_A, KEY_A), (ISSUER_B, AUDIENCE_B, KEY_B))
    return client


def _submit(client: TestClient, headers: dict[str, str], ticker: str = "AAPL", **extra: Any):
    return client.post("/analyze/async", json={"ticker": ticker}, headers={**headers, **extra})


def _guest(sub: str | None = None, **overrides: Any) -> dict[str, str]:
    return bearer(mint(sub=sub or "guest:" + str(uuid.uuid4()), **overrides))


def _analyst(sub: str) -> dict[str, str]:
    return bearer(mint(sub=sub, role="analyst"))


def _admin_token() -> dict[str, str]:
    return bearer(mint(KEY_B, iss=ISSUER_B, aud=AUDIENCE_B, sub="ops-1", role="admin"))


def _assert_unauthorized(response) -> None:
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"


# --------------------------------------------------------------------------- configuration


def test_jwt_issuers_defaults_to_an_empty_allowlist(monkeypatch):
    """test-plan §20 Issuer allowlist: with nothing configured the allowlist is empty."""
    monkeypatch.delenv("JWT_ISSUERS", raising=False)
    assert Settings(_env_file=None).jwt_issuers == []


def test_jwt_issuers_env_is_a_json_list_of_issuer_models(monkeypatch):
    """test-plan §20 Issuer allowlist: JWT_ISSUERS (JSON list) configures issuer, audience, key."""
    entry = {"issuer": ISSUER_A, "audience": AUDIENCE_A, "public_key_pem": KEY_A.public_pem}
    monkeypatch.setenv("JWT_ISSUERS", json.dumps([entry]))
    issuers = Settings(_env_file=None).jwt_issuers
    assert [(i.issuer, i.audience, i.public_key_pem) for i in issuers] == [
        (ISSUER_A, AUDIENCE_A, KEY_A.public_pem)
    ]


@pytest.mark.parametrize(
    "entry",
    [
        {"issuer": ISSUER_A, "audience": AUDIENCE_A, "public_key_pem": "k", "algorithm": "HS256"},
        {"issuer": ISSUER_A, "audience": AUDIENCE_A},
        {"issuer": ISSUER_A, "public_key_pem": "k"},
        {"audience": AUDIENCE_A, "public_key_pem": "k"},
    ],
    ids=["extra-field", "no-key", "no-audience", "no-issuer"],
)
def test_an_issuer_entry_with_extra_or_missing_fields_is_refused(entry: dict[str, str]):
    """test-plan §20 Issuer allowlist: the issuer model forbids extra fields and needs all three."""
    from app.config import JwtIssuer

    with pytest.raises(ValidationError):
        JwtIssuer(**entry)


# --------------------------------------------------------------------------- development issuer


@pytest.mark.parametrize(
    "headers",
    [ADMIN_KEY_HEADERS, bearer(API_KEY)],
    ids=["x-api-key", "bearer"],
)
@pytest.mark.parametrize("path", ["/stats", "/metrics", "/ingest/AAPL"])
def test_the_development_api_key_is_still_an_admin(client: TestClient, headers, path):
    """test-plan §12 S7.2 'API auth' + §20 Issuer allowlist: API_KEY stays the dev issuer (admin)."""
    assert client.get(path, headers=headers).status_code == 200


def test_the_development_api_key_can_ingest_and_analyze(client: TestClient):
    """test-plan §12 S7.2 'API auth': the dev-issuer admin may ingest and submit analyses."""
    assert (
        client.post("/ingest", json={"ticker": "AAPL"}, headers=ADMIN_KEY_HEADERS).status_code
        == 200
    )
    assert (
        client.post("/analyze", json={"ticker": "AAPL"}, headers=ADMIN_KEY_HEADERS).status_code
        == 200
    )


def test_a_wrong_api_key_is_a_typed_401(client: TestClient):
    """test-plan §12 S7.2 'API auth': an invalid key is 401 with the typed envelope."""
    _assert_unauthorized(client.get("/jobs", headers=bearer("wrong")))
    _assert_unauthorized(client.get("/jobs", headers={"X-API-Key": "wrong"}))


def test_no_credentials_is_a_typed_401(client: TestClient):
    """test-plan §12 S7.2 'API auth': a missing credential is 401."""
    _assert_unauthorized(client.get("/jobs"))


def test_nothing_configured_fails_closed_with_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """test-plan §12 S7.2 'API auth': no api_key and no issuers is 503 auth_unconfigured."""
    monkeypatch.setattr(api_main.settings, "api_key", None)
    for headers in (ADMIN_KEY_HEADERS, bearer(mint()), {}):
        response = client.get("/jobs", headers=headers)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "auth_unconfigured"


def test_an_empty_allowlist_refuses_every_jwt(client: TestClient):
    """test-plan §20 Issuer allowlist: with no issuer configured, a well-formed token is refused."""
    for role in ("guest", "analyst", "admin"):
        _assert_unauthorized(client.get("/jobs", headers=bearer(mint(role=role))))


def test_an_allowlist_alone_authenticates_without_an_api_key(
    allowlisted: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """test-plan §20 Issuer allowlist: the API key is optional once an issuer is configured."""
    monkeypatch.setattr(api_main.settings, "api_key", None)
    assert allowlisted.get("/jobs", headers=_guest()).status_code == 200
    _assert_unauthorized(allowlisted.get("/jobs", headers=bearer(mint(KEY_UNLISTED))))


# --------------------------------------------------------------------------- issuer allowlist


@pytest.mark.parametrize("role", ["guest", "analyst", "admin"])
def test_a_valid_token_of_each_role_is_accepted(allowlisted: TestClient, role: str):
    """test-plan §20 Issuer allowlist / Guest access: a valid ES256 token of any role signs in."""
    headers = bearer(mint(role=role))
    assert allowlisted.get("/jobs", headers=headers).status_code == 200
    assert allowlisted.post("/analyze", json={"ticker": "AAPL"}, headers=headers).status_code == 200
    assert _submit(allowlisted, headers).status_code == 202


def test_each_allowlisted_issuer_is_verified_with_its_own_key_and_audience(
    allowlisted: TestClient,
):
    """test-plan §20 Issuer allowlist: a second allowlisted issuer signs in with its own key."""
    assert allowlisted.get("/stats", headers=_admin_token()).status_code == 200


REFUSALS = {
    "foreign-issuer": dict(iss="https://evil.example/guest"),
    "wrong-audience": dict(aud="some-other-api"),
    "expired": dict(exp=int(time.time()) - 3600),
    "missing-exp": dict(exp=MISSING),
    "missing-sub": dict(sub=MISSING),
    "missing-iss": dict(iss=MISSING),
    "missing-aud": dict(aud=MISSING),
    "missing-role": dict(role=MISSING),
    "unknown-role": dict(role="superuser"),
    "role-case": dict(role="ADMIN"),
    "role-not-a-string": dict(role=["admin"]),
    "this-issuers-audience-for-the-other-issuer": dict(aud=AUDIENCE_B),
}


@pytest.mark.parametrize("overrides", REFUSALS.values(), ids=REFUSALS.keys())
def test_a_token_failing_one_check_is_refused(allowlisted: TestClient, overrides):
    """test-plan §20 Issuer allowlist: foreign issuer, wrong audience, expired, or missing or
    invalid claims are each a typed 401."""
    _assert_unauthorized(allowlisted.get("/jobs", headers=bearer(mint(**overrides))))


def test_the_baseline_token_the_refusals_vary_is_accepted(allowlisted: TestClient):
    """test-plan §20 Issuer allowlist: control for the refusal cases — one claim away from valid."""
    assert allowlisted.get("/jobs", headers=bearer(mint())).status_code == 200


def _signed_by_wrong_key() -> str:
    return mint(KEY_UNLISTED)


def _issuer_a_signed_by_issuer_b_key() -> str:
    return mint(KEY_B)


def _issuer_b_signed_by_issuer_a_key() -> str:
    return mint(KEY_A, iss=ISSUER_B, aud=AUDIENCE_B)


def _alg_none() -> str:
    return forge({"alg": "none", "typ": "JWT"}, _claims(role="admin"), None)


def _hs256_with_the_public_pem_as_secret() -> str:
    return forge({"alg": "HS256", "typ": "JWT"}, _claims(role="admin"), KEY_A.public_pem.encode())


def _payload_swapped_after_signing() -> str:
    header, _, signature = mint(role="guest").split(".")
    return ".".join([header, _b64(json.dumps(_claims(role="admin")).encode()), signature])


FORGERIES = {
    "signature-by-an-unlisted-key": _signed_by_wrong_key,
    "issuer-a-token-signed-by-issuer-b-key": _issuer_a_signed_by_issuer_b_key,
    "issuer-b-token-signed-by-issuer-a-key": _issuer_b_signed_by_issuer_a_key,
    "alg-none": _alg_none,
    "hs256-confusion": _hs256_with_the_public_pem_as_secret,
    "payload-swapped-after-signing": _payload_swapped_after_signing,
    "not-a-jwt": lambda: "not-a-jwt",
    "empty-segments": lambda: "..",
}


@pytest.mark.parametrize("make_token", FORGERIES.values(), ids=FORGERIES.keys())
def test_a_forged_or_misattributed_token_is_refused(allowlisted: TestClient, make_token):
    """test-plan §20 Issuer allowlist: only ES256 verified by the key of the token's own issuer
    is accepted; none, HS256 confusion, foreign keys and tampering are 401."""
    _assert_unauthorized(allowlisted.get("/jobs", headers=bearer(make_token())))


def test_a_refused_token_leaves_no_run_behind(allowlisted: TestClient):
    """test-plan §20 Issuer allowlist: a refused submission creates nothing."""
    response = _submit(allowlisted, bearer(mint(exp=int(time.time()) - 3600)))
    _assert_unauthorized(response)
    assert allowlisted.get("/jobs", headers=ADMIN_KEY_HEADERS).json() == []


# --------------------------------------------------------------------------- admin-only routes

ADMIN_ROUTES = [
    ("get", "/stats", None),
    ("get", "/metrics", None),
    ("post", "/ingest", {"ticker": "AAPL"}),
    ("get", "/ingest/AAPL", None),
]


def _call(client: TestClient, method: str, path: str, body, headers):
    if body is None:
        return getattr(client, method)(path, headers=headers)
    return getattr(client, method)(path, json=body, headers=headers)


@pytest.mark.parametrize("role", ["guest", "analyst"])
@pytest.mark.parametrize(("method", "path", "body"), ADMIN_ROUTES)
def test_guest_and_analyst_are_forbidden_on_admin_routes(
    allowlisted: TestClient, role: str, method, path, body
):
    """test-plan §20 Guest access: a valid guest token is refused on admin routes (403, not 401);
    ADR-0010 §5: analyst likewise."""
    response = _call(allowlisted, method, path, body, bearer(mint(role=role)))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


@pytest.mark.parametrize(("method", "path", "body"), ADMIN_ROUTES)
def test_an_admin_token_reaches_the_admin_routes(allowlisted: TestClient, method, path, body):
    """test-plan §20 Customer isolation (admin) : an admin token passes the admin-only routes."""
    assert _call(allowlisted, method, path, body, _admin_token()).status_code == 200


def test_a_forbidden_ingest_does_not_ingest(allowlisted: TestClient, monkeypatch):
    """test-plan §20 Guest access: a refused /ingest has no effect (the ingest stub would
    otherwise make the corpus change visible)."""
    ingested: list[str] = []

    async def recording_ingest(ticker: str, replace_existing: bool = False):
        ingested.append(ticker)
        return await fake_ingest_10k_for_ticker(ticker, replace_existing)

    monkeypatch.setattr(api_main, "ingest_10k_for_ticker", recording_ingest)
    allowlisted.post("/ingest", json={"ticker": "AAPL"}, headers=_guest())
    assert ingested == []
    allowlisted.post("/ingest", json={"ticker": "AAPL"}, headers=_admin_token())
    assert ingested == ["AAPL"]


# --------------------------------------------------------------------------- job isolation


def _job_id(client: TestClient, headers: dict[str, str], ticker: str = "AAPL") -> str:
    response = _submit(client, headers, ticker)
    assert response.status_code == 202
    return str(response.json()["job_id"])


def _scrub(response, job_id: str) -> tuple[int, str, str]:
    error = response.json()["error"]
    return response.status_code, error["code"], error["message"].replace(job_id, "<job>")


@pytest.mark.parametrize("reader_role", ["guest", "analyst"])
def test_another_customers_job_reads_exactly_like_a_missing_job(
    allowlisted: TestClient, reader_role: str
):
    """test-plan §20 Customer isolation / Guest access: a guest token cannot read another guest's
    job, and the refusal does not reveal that the job exists."""
    owner = _guest("guest:owner")
    job_id = _job_id(allowlisted, owner)
    other = bearer(mint(sub="guest:other", role=reader_role))

    foreign = allowlisted.get(f"/jobs/{job_id}", headers=other)
    missing = allowlisted.get(f"/jobs/{NOT_FOUND}", headers=other)

    assert foreign.status_code == 404
    assert _scrub(foreign, job_id) == _scrub(missing, NOT_FOUND)
    assert "AAPL" not in foreign.text


def test_the_owner_reads_the_job_across_tokens_of_the_same_subject(allowlisted: TestClient):
    """test-plan §20 Customer isolation: the customer is the token's `sub`, not the token — a
    fresh 15-minute token of the same guest reads the same job."""
    first_token = _guest("guest:owner")
    second_token = _guest("guest:owner")
    assert first_token != second_token
    job_id = _job_id(allowlisted, first_token)
    for headers in (first_token, second_token):
        response = allowlisted.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id


def test_an_analyst_reads_only_its_own_job(allowlisted: TestClient):
    """test-plan §20 Customer isolation: `analyst` reads its own job, not another analyst's."""
    job_id = _job_id(allowlisted, _analyst("analyst:one"))
    assert allowlisted.get(f"/jobs/{job_id}", headers=_analyst("analyst:one")).status_code == 200
    assert allowlisted.get(f"/jobs/{job_id}", headers=_analyst("analyst:two")).status_code == 404


def test_an_admin_reads_any_customers_job(allowlisted: TestClient):
    """test-plan §20 Customer isolation: `admin` reads any job (admin token and dev API key)."""
    job_id = _job_id(allowlisted, _guest("guest:owner"))
    for headers in (_admin_token(), ADMIN_KEY_HEADERS, bearer(API_KEY)):
        response = allowlisted.get(f"/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id


def test_a_guest_cannot_read_a_job_the_development_admin_submitted(allowlisted: TestClient):
    """test-plan §20 Customer isolation: jobs submitted with the dev API key belong to customer
    `admin`, which no guest is."""
    job_id = _job_id(allowlisted, ADMIN_KEY_HEADERS)
    assert allowlisted.get(f"/jobs/{job_id}", headers=_guest()).status_code == 404


def test_a_job_poll_still_needs_a_valid_credential(allowlisted: TestClient):
    """test-plan §20 Customer isolation: an unauthenticated poll is 401, not 404."""
    job_id = _job_id(allowlisted, _guest("guest:owner"))
    _assert_unauthorized(allowlisted.get(f"/jobs/{job_id}"))
    _assert_unauthorized(allowlisted.get(f"/jobs/{job_id}", headers=bearer(mint(KEY_UNLISTED))))


def _listed(client: TestClient, headers: dict[str, str]) -> list[str]:
    response = client.get("/jobs", headers=headers)
    assert response.status_code == 200
    return [run["job_id"] for run in response.json()]


def test_the_job_list_is_scoped_to_the_caller_and_global_for_admin(allowlisted: TestClient):
    """test-plan §20 Customer isolation: `analyst`/guest see only own runs in GET /jobs; `admin`
    sees all."""
    one = _job_id(allowlisted, _guest("guest:one"))
    two = _job_id(allowlisted, _guest("guest:two"), "MSFT")
    analyst = _job_id(allowlisted, _analyst("analyst:one"))
    dev = _job_id(allowlisted, ADMIN_KEY_HEADERS, "MSFT")

    assert _listed(allowlisted, _guest("guest:one")) == [one]
    assert _listed(allowlisted, _guest("guest:two")) == [two]
    assert _listed(allowlisted, _analyst("analyst:one")) == [analyst]
    assert _listed(allowlisted, _guest("guest:nobody")) == []
    expected = {one, two, analyst, dev}
    assert set(_listed(allowlisted, _admin_token())) == expected
    assert set(_listed(allowlisted, ADMIN_KEY_HEADERS)) == expected


def test_the_scoped_list_honours_the_limit_after_scoping(allowlisted: TestClient):
    """test-plan §20 Customer isolation: `limit` counts the caller's runs, not other customers'."""
    mine = _job_id(allowlisted, _guest("guest:one"))
    for _ in range(3):
        _job_id(allowlisted, _guest("guest:two"), "MSFT")
    response = allowlisted.get("/jobs", params={"limit": 1}, headers=_guest("guest:one"))
    assert [run["job_id"] for run in response.json()] == [mine]


def test_a_run_stored_before_identity_is_visible_to_admin_only(
    allowlisted: TestClient,
):
    """test-plan §20 Customer isolation: runs stored before S7.2 carry no customer and are
    visible to `admin` only."""
    legacy = str(uuid.uuid4())
    api_main.run_store.create_run(legacy, "AAPL")
    api_main.run_store.mark_failed(legacy, "stored before identity")

    for headers in (_guest("guest:one"), _analyst("analyst:one")):
        foreign = allowlisted.get(f"/jobs/{legacy}", headers=headers)
        missing = allowlisted.get(f"/jobs/{NOT_FOUND}", headers=headers)
        assert foreign.status_code == 404
        assert _scrub(foreign, legacy) == _scrub(missing, NOT_FOUND)
        assert legacy not in _listed(allowlisted, headers)

    for headers in (_admin_token(), ADMIN_KEY_HEADERS):
        assert allowlisted.get(f"/jobs/{legacy}", headers=headers).status_code == 200
        assert legacy in _listed(allowlisted, headers)


# --------------------------------------------------------------------------- idempotency


def test_the_same_key_from_two_customers_makes_two_jobs(allowlisted: TestClient):
    """test-plan §12 S7.2 'Idempotency' + §20 Customer isolation: idempotency keys are scoped per
    customer; each customer replays its own key to its own job."""
    key = {"Idempotency-Key": "shared-key"}
    a_headers, b_headers = _guest("guest:a"), _guest("guest:b")
    first_a = _submit(allowlisted, a_headers, **key)
    first_b = _submit(allowlisted, b_headers, **key)
    assert first_a.status_code == first_b.status_code == 202
    assert first_a.json()["job_id"] != first_b.json()["job_id"]

    again_a = _submit(allowlisted, _guest("guest:a"), **key)
    again_b = _submit(allowlisted, _guest("guest:b"), **key)
    assert again_a.json()["job_id"] == first_a.json()["job_id"]
    assert again_b.json()["job_id"] == first_b.json()["job_id"]
    assert again_b.headers["location"] == first_b.headers["location"]


def test_reusing_another_customers_key_with_a_different_body_is_not_a_conflict(
    allowlisted: TestClient,
):
    """test-plan §20 Customer isolation: a caller cannot reuse another customer's idempotency
    key — and a 409 would reveal the key exists."""
    key = {"Idempotency-Key": "shared-key"}
    mine = _submit(allowlisted, _guest("guest:a"), "AAPL", **key)
    theirs = _submit(allowlisted, _guest("guest:b"), "MSFT", **key)
    assert theirs.status_code == 202
    assert theirs.json()["job_id"] != mine.json()["job_id"]
    assert allowlisted.get(theirs.headers["location"], headers=_guest("guest:b")).status_code == 200
    assert allowlisted.get(mine.headers["location"], headers=_guest("guest:b")).status_code == 404


def test_the_same_customer_replays_across_tokens_and_still_conflicts_on_a_changed_body(
    allowlisted: TestClient,
):
    """test-plan §12 S7.2 'Idempotency': the key belongs to the customer (`sub`), not to the
    short-lived token that sent it."""
    key = {"Idempotency-Key": "request-1"}
    first = _submit(allowlisted, _guest("guest:a"), "AAPL", **key)
    same = _submit(allowlisted, _guest("guest:a"), "AAPL", **key)
    changed = _submit(allowlisted, _guest("guest:a"), "MSFT", **key)
    assert first.status_code == same.status_code == 202
    assert first.json()["job_id"] == same.json()["job_id"]
    assert changed.status_code == 409


def test_the_development_admin_and_a_guest_do_not_share_keys(allowlisted: TestClient):
    """test-plan §20 Customer isolation: the dev issuer's customer `admin` is a customer like any
    other for idempotency."""
    key = {"Idempotency-Key": "shared-key"}
    admin = _submit(allowlisted, ADMIN_KEY_HEADERS, **key)
    guest = _submit(allowlisted, _guest("guest:a"), **key)
    assert admin.json()["job_id"] != guest.json()["job_id"]
    assert _submit(allowlisted, bearer(API_KEY), **key).json()["job_id"] == admin.json()["job_id"]


# --------------------------------------------------------------------------- rate limit


def test_the_submission_rate_limit_is_per_customer(
    allowlisted: TestClient, monkeypatch: pytest.MonkeyPatch
):
    """test-plan §12 S7.2 'Rate limit': the limit follows the customer (`sub`), not the token and
    not the whole API — one guest's burst does not throttle another."""
    monkeypatch.setattr(api_main.settings, "api_rate_limit", 1)
    assert (
        allowlisted.post("/analyze", json={"ticker": "AAPL"}, headers=_guest("guest:a")).status_code
        == 200
    )

    throttled = allowlisted.post("/analyze", json={"ticker": "AAPL"}, headers=_guest("guest:a"))
    assert throttled.status_code == 429
    assert int(throttled.headers["retry-after"]) >= 1

    for headers in (_guest("guest:b"), _analyst("analyst:a"), ADMIN_KEY_HEADERS):
        response = allowlisted.post("/analyze", json={"ticker": "AAPL"}, headers=headers)
        assert response.status_code == 200
