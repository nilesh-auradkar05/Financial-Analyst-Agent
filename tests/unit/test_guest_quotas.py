"""S7.2 slice (c): guest quotas (ADR-0010 amendment 2026-10-11 §2).

Trace: docs/test-plan.md §20 row "Guest quotas": the second live job in one guest session, the
fourth from one IP hash in a UTC day and the 21st across all guests are refused with 429 and
`Retry-After` (the next UTC midnight for the daily caps). The example-memo clause is S7.9.
The `ip_hash` claim (a guest token without one is a 401) has no test-plan row of its own yet; it
is exercised here under the same row (ADR-0010 amendment §2: the IP is kept only as a hash).

Everything is exercised through HTTP routes. Contract seams the tests rely on:
  * `app.main.utc_now() -> datetime` (timezone-aware UTC): the only clock the quotas read;
  * `app.main.GuestQuotas` (zero-argument constructor) and the module attribute
    `app.main.guest_quotas`, replaced per test like `submission_limiter`.
Runs that are "live" are created through the run store's public methods, because the stubbed
agent finishes a job before the submitting request returns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as api_main
from tests.unit.test_identity import (  # noqa: F401  (fixtures are re-exported for pytest)
    ADMIN_KEY_HEADERS,
    AUDIENCE_B,
    ISSUER_B,
    KEY_B,
    MISSING,
    allowlisted,
    bearer,
    client,
    mint,
)

UTC = timezone.utc
NOON = datetime(2026, 11, 23, 13, 45, 30, tzinfo=UTC)
SUBMIT_PATHS = ["/analyze/async", "/analyze"]
IP_CAP = 3
GLOBAL_CAP = 20


@dataclass
class FakeClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now


def next_midnight(moment: datetime) -> datetime:
    return datetime.combine(moment.date() + timedelta(days=1), time(), tzinfo=UTC)


def seconds_to_midnight(moment: datetime) -> int:
    return int((next_midnight(moment) - moment).total_seconds())


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock(NOON)
    monkeypatch.setattr(api_main, "utc_now", fake)
    return fake


@pytest.fixture()
def api(
    allowlisted: TestClient,  # noqa: F811
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """An allow-listed API with a fake UTC clock and a rate limit that never gets in the way."""
    monkeypatch.setattr(api_main.settings, "api_rate_limit", 1000)
    return allowlisted


def fresh_ip() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex


def guest(sub: str | None = None, ip: str | None = None, **overrides: Any) -> dict[str, str]:
    return bearer(
        mint(sub=sub or "guest:" + str(uuid.uuid4()), ip_hash=ip or fresh_ip(), **overrides)
    )


def analyst(sub: str = "analyst:one") -> dict[str, str]:
    return bearer(mint(sub=sub, role="analyst"))


def admin_token() -> dict[str, str]:
    return bearer(mint(KEY_B, iss=ISSUER_B, aud=AUDIENCE_B, sub="ops-1", role="admin"))


def submit(
    http: TestClient,
    headers: dict[str, str],
    path: str = "/analyze/async",
    ticker: str = "AAPL",
    **extra: str,
):
    return http.post(path, json={"ticker": ticker}, headers={**headers, **extra})


def start_live_job(sub: str, *, running: bool = False) -> str:
    """A job of customer `sub` that is still in flight, made through the run store's public API."""
    job_id = str(uuid.uuid4())
    api_main.run_store.create_run(job_id, "AAPL", customer_id=sub)
    if running:
        api_main.run_store.mark_running(job_id)
    return job_id


def run_ids(http: TestClient) -> set[str]:
    response = http.get("/jobs", headers=ADMIN_KEY_HEADERS)
    assert response.status_code == 200
    return {run["job_id"] for run in response.json()}


def assert_quota_refusal(response, retry_after: int) -> None:
    assert response.status_code == 429
    assert response.headers["retry-after"] == str(retry_after)
    error = response.json()["error"]
    assert error["code"] == "quota_exceeded"
    assert set(error) == {"code", "message", "error_id"}


def admit_from(http: TestClient, ip: str, count: int) -> list[str]:
    """`count` submissions from `count` different guest sessions behind one IP hash."""
    job_ids = []
    for _ in range(count):
        response = submit(http, guest(ip=ip))
        assert response.status_code == 202
        job_ids.append(response.json()["job_id"])
    return job_ids


def admit_across_ips(http: TestClient, count: int) -> None:
    for _ in range(count):
        assert submit(http, guest()).status_code == 202


# --------------------------------------------------------------------------- ip_hash claim


@pytest.mark.parametrize(
    "ip_hash",
    [MISSING, "", None, 12345, ["a" * 64], {"hash": "a" * 64}],
    ids=["missing", "empty", "null", "number", "list", "object"],
)
@pytest.mark.parametrize("path", ["/jobs", *SUBMIT_PATHS])
def test_a_guest_token_without_a_usable_ip_hash_is_a_401(api: TestClient, path: str, ip_hash):
    """test-plan §20 Guest quotas (IP hash claim): the quota key must come from the token, so a
    guest token without a non-empty string `ip_hash` is a typed 401 and creates no run."""
    headers = bearer(mint(ip_hash=ip_hash))
    if path == "/jobs":
        response = api.get(path, headers=headers)
    else:
        response = submit(api, headers, path)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"
    assert run_ids(api) == set()


@pytest.mark.parametrize("path", ["/jobs", *SUBMIT_PATHS])
def test_a_guest_token_with_an_ip_hash_is_accepted(api: TestClient, path: str):
    """test-plan §20 Guest quotas (IP hash claim): control for the 401 cases — one claim away."""
    headers = guest()
    response = api.get(path, headers=headers) if path == "/jobs" else submit(api, headers, path)
    assert response.status_code in {200, 202}


@pytest.mark.parametrize("role", ["analyst", "admin"])
def test_only_guest_tokens_need_an_ip_hash(api: TestClient, role: str):
    """test-plan §20 Guest quotas: the IP hash is a guest-session claim; analyst and admin tokens
    sign in without it."""
    if role == "admin":
        headers = bearer(
            mint(
                KEY_B,
                iss=ISSUER_B,
                aud=AUDIENCE_B,
                sub="ops-1",
                role="admin",
                ip_hash=MISSING,
            )
        )
    else:
        headers = bearer(mint(sub="analyst:one", role="analyst", ip_hash=MISSING))
    assert api.get("/jobs", headers=headers).status_code == 200
    assert submit(api, headers).status_code == 202


# --------------------------------------------------------------------------- live-job cap


@pytest.mark.parametrize("path", SUBMIT_PATHS)
@pytest.mark.parametrize("running", [False, True], ids=["pending", "running"])
def test_a_second_live_job_in_one_guest_session_is_refused(
    api: TestClient, path: str, running: bool
):
    """test-plan §20 Guest quotas: with one job pending or running, the guest's next submission
    is 429 `quota_exceeded` with `Retry-After: 60`, and creates no run."""
    sub = "guest:busy"
    live = start_live_job(sub, running=running)
    before = run_ids(api)

    response = submit(api, guest(sub), path)

    assert_quota_refusal(response, 60)
    assert run_ids(api) == before == {live}


def test_the_live_cap_follows_the_session_not_the_token(api: TestClient):
    """test-plan §20 Guest quotas: a fresh 15-minute token of the same guest (same `sub`, even
    from another IP hash) is the same session and is refused too."""
    start_live_job("guest:busy")
    for headers in (guest("guest:busy"), guest("guest:busy", ip=fresh_ip())):
        assert_quota_refusal(submit(api, headers), 60)


def test_another_guests_live_job_does_not_block_this_guest(api: TestClient):
    """test-plan §20 Guest quotas: the live cap counts the caller's own jobs only."""
    start_live_job("guest:busy")
    assert submit(api, guest("guest:other")).status_code == 202


@pytest.mark.parametrize(
    "finish",
    [
        lambda store, job_id: store.mark_failed(job_id, "boom"),
        lambda store, job_id: store.mark_completed(job_id, {"investment_memo": "m"}),
        lambda store, job_id: store.mark_finished(job_id, "completed", {"investment_memo": "m"}),
        lambda store, job_id: store.mark_finished(job_id, "degraded", {"investment_memo": "m"}),
        lambda store, job_id: store.mark_finished(
            job_id, "evidence_missing", {"investment_memo": "m"}
        ),
    ],
    ids=["failed", "completed", "completed-final", "degraded", "evidence_missing"],
)
@pytest.mark.parametrize("path", SUBMIT_PATHS)
def test_a_terminal_job_frees_the_guests_live_slot(api: TestClient, path: str, finish):
    """test-plan §20 Guest quotas: once the live job is terminal, whatever its status, the guest
    is admitted again (the daily caps still apply)."""
    sub = "guest:busy"
    live = start_live_job(sub, running=True)
    assert submit(api, guest(sub), path).status_code == 429

    finish(api_main.run_store, live)

    assert submit(api, guest(sub), path).status_code in {200, 202}


def test_analysts_and_admins_are_not_held_to_one_live_job(api: TestClient):
    """test-plan §20 Guest quotas: only role `guest` is limited; an analyst, an admin token and
    the development API key submit while their own job is still live."""
    start_live_job("analyst:one")
    start_live_job("ops-1")
    start_live_job("admin")
    assert submit(api, analyst("analyst:one")).status_code == 202
    assert submit(api, admin_token()).status_code == 202
    assert submit(api, ADMIN_KEY_HEADERS).status_code == 202


# --------------------------------------------------------------------------- IP cap


@pytest.mark.parametrize("path", SUBMIT_PATHS)
def test_the_fourth_job_from_one_ip_hash_in_a_utc_day_is_refused(
    api: TestClient, clock: FakeClock, path: str
):
    """test-plan §20 Guest quotas: three guest sessions behind one IP hash are admitted; the
    fourth is 429 with `Retry-After` = seconds to the next UTC midnight, and creates no run."""
    ip = fresh_ip()
    for _ in range(IP_CAP):
        assert submit(api, guest(ip=ip), path).status_code in {200, 202}
    before = run_ids(api)

    response = submit(api, guest(ip=ip), path)

    assert_quota_refusal(response, seconds_to_midnight(clock.now))
    assert run_ids(api) == before


def test_another_ip_hash_is_not_affected_by_a_full_one(api: TestClient):
    """test-plan §20 Guest quotas: the IP cap is per hash."""
    ip = fresh_ip()
    admit_from(api, ip, IP_CAP)
    assert submit(api, guest(ip=ip)).status_code == 429
    assert submit(api, guest(ip=fresh_ip())).status_code == 202


def test_one_guest_session_spends_the_ip_cap_with_finished_jobs(api: TestClient):
    """test-plan §20 Guest quotas: the IP cap counts admitted jobs, not live ones — a single
    session that finishes each job still cannot submit a fourth from the same IP hash."""
    sub, ip = "guest:patient", fresh_ip()
    for _ in range(IP_CAP):
        response = submit(api, guest(sub, ip))
        assert response.status_code == 202
    assert_quota_refusal(submit(api, guest(sub, ip)), seconds_to_midnight(NOON))


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 11, 23, 0, 0, 0, tzinfo=UTC),
        NOON,
        datetime(2026, 11, 23, 23, 59, 59, tzinfo=UTC),
        datetime(2026, 12, 31, 23, 0, 0, tzinfo=UTC),
    ],
    ids=["midnight", "afternoon", "last-second", "year-end"],
)
def test_retry_after_is_whole_seconds_to_the_next_utc_midnight(
    api: TestClient, clock: FakeClock, moment: datetime
):
    """test-plan §20 Guest quotas: `Retry-After` for a daily cap is the whole seconds until the
    next UTC midnight (86400 at midnight, 1 in the last second)."""
    clock.now = moment
    ip = fresh_ip()
    admit_from(api, ip, IP_CAP)

    response = submit(api, guest(ip=ip))

    assert_quota_refusal(response, seconds_to_midnight(moment))
    assert response.headers["retry-after"].isdigit()


def test_the_ip_cap_resets_at_the_utc_day_boundary(api: TestClient, clock: FakeClock):
    """test-plan §20 Guest quotas: the count is per UTC day — refused in the last second of a day,
    admitted again from its first second, and then capped afresh at three."""
    clock.now = datetime(2026, 11, 23, 23, 59, 58, tzinfo=UTC)
    ip = fresh_ip()
    admit_from(api, ip, IP_CAP)
    clock.now = datetime(2026, 11, 23, 23, 59, 59, tzinfo=UTC)
    assert_quota_refusal(submit(api, guest(ip=ip)), 1)

    clock.now = datetime(2026, 11, 24, 0, 0, 0, tzinfo=UTC)
    admit_from(api, ip, IP_CAP)
    assert_quota_refusal(submit(api, guest(ip=ip)), 86400)


# --------------------------------------------------------------------------- global cap


@pytest.mark.parametrize("path", SUBMIT_PATHS)
def test_the_twenty_first_guest_job_of_a_utc_day_is_refused(
    api: TestClient, clock: FakeClock, path: str
):
    """test-plan §20 Guest quotas: twenty guest jobs from twenty IP hashes are admitted; the
    twenty-first, from any guest, is 429 with `Retry-After` to the next UTC midnight."""
    admit_across_ips(api, GLOBAL_CAP)
    before = run_ids(api)

    response = submit(api, guest(), path)

    assert_quota_refusal(response, seconds_to_midnight(clock.now))
    assert len(before) == GLOBAL_CAP
    assert run_ids(api) == before


def test_the_global_cap_resets_at_the_utc_day_boundary(api: TestClient, clock: FakeClock):
    """test-plan §20 Guest quotas: the global count restarts with the UTC day."""
    clock.now = datetime(2026, 11, 23, 23, 59, 59, tzinfo=UTC)
    admit_across_ips(api, GLOBAL_CAP)
    assert_quota_refusal(submit(api, guest()), 1)

    clock.now = datetime(2026, 11, 24, 0, 0, 0, tzinfo=UTC)
    admit_across_ips(api, GLOBAL_CAP)
    assert_quota_refusal(submit(api, guest()), 86400)


def test_analyst_and_admin_jobs_do_not_spend_the_guest_quotas(api: TestClient):
    """test-plan §20 Guest quotas: only role `guest` is limited and counted — 25 submissions each
    by an analyst, an admin token and the development key leave all 20 guest slots free."""
    for _ in range(25):
        assert submit(api, analyst()).status_code == 202
        assert submit(api, admin_token()).status_code == 202
        assert submit(api, ADMIN_KEY_HEADERS).status_code == 202
    admit_across_ips(api, GLOBAL_CAP)
    assert submit(api, guest()).status_code == 429


# --------------------------------------------------------------------------- refusals consume nothing


def test_refused_submissions_do_not_spend_the_ip_cap(api: TestClient):
    """test-plan §20 Guest quotas: a refusal consumes no quota — four refusals on the live-job cap
    leave the IP with its three admissions."""
    ip = fresh_ip()
    live = start_live_job("guest:busy")
    for _ in range(4):
        assert submit(api, guest("guest:busy", ip)).status_code == 429
    api_main.run_store.mark_failed(live, "done")

    assert submit(api, guest("guest:busy", ip)).status_code == 202
    admit_from(api, ip, IP_CAP - 1)
    assert submit(api, guest(ip=ip)).status_code == 429


def test_refused_submissions_do_not_spend_the_global_cap(api: TestClient):
    """test-plan §20 Guest quotas: refusals on the IP cap leave the global count at the number of
    jobs actually admitted, so exactly twenty guest jobs fit in the day."""
    ip = fresh_ip()
    admit_from(api, ip, IP_CAP)
    for _ in range(5):
        assert submit(api, guest(ip=ip)).status_code == 429

    admit_across_ips(api, GLOBAL_CAP - IP_CAP)

    assert submit(api, guest()).status_code == 429
    assert len(run_ids(api)) == GLOBAL_CAP


# --------------------------------------------------------------------------- idempotent replay


def test_an_idempotent_replay_returns_the_job_and_spends_no_quota(api: TestClient):
    """test-plan §20 Guest quotas + §12 S7.2 'Idempotency': replaying a key returns the existing
    job without a new run and without using IP or global quota, so a third and no fourth session
    still fits behind the IP."""
    ip = fresh_ip()
    firsts = {}
    for name in ("one", "two"):
        sub = f"guest:{name}"
        response = submit(api, guest(sub, ip), **{"Idempotency-Key": f"key-{name}"})
        assert response.status_code == 202
        firsts[sub] = response.json()["job_id"]

    for _ in range(3):
        for sub, job_id in firsts.items():
            name = sub.split(":")[1]
            again = submit(api, guest(sub, ip), **{"Idempotency-Key": f"key-{name}"})
            assert again.status_code == 202
            assert again.json()["job_id"] == job_id

    assert len(run_ids(api)) == 2
    assert submit(api, guest(ip=ip)).status_code == 202
    assert submit(api, guest(ip=ip)).status_code == 429


def test_an_idempotent_replay_is_served_even_when_the_ip_cap_is_full(api: TestClient):
    """test-plan §20 Guest quotas + §12 S7.2 'Idempotency': a replay is not a new job, so it is
    not refused by a cap that new jobs from the same IP hash hit."""
    ip, sub = fresh_ip(), "guest:first"
    first = submit(api, guest(sub, ip), **{"Idempotency-Key": "key-1"})
    admit_from(api, ip, IP_CAP - 1)
    assert submit(api, guest(ip=ip)).status_code == 429

    replay = submit(api, guest(sub, ip), **{"Idempotency-Key": "key-1"})

    assert replay.status_code == 202
    assert replay.json()["job_id"] == first.json()["job_id"]


# --------------------------------------------------------------------------- refusal body


def test_a_refusal_discloses_nothing_about_the_caller(api: TestClient):
    """test-plan §20 Guest quotas: the 429 body is the typed envelope and carries no IP hash,
    token or customer id, whichever cap refused."""
    ip, sub = fresh_ip(), "guest:busy"
    start_live_job(sub)
    token = mint(sub=sub, ip_hash=ip)
    live_refusal = api.post("/analyze/async", json={"ticker": "AAPL"}, headers=bearer(token))
    admit_from(api, ip, IP_CAP)
    ip_refusal = submit(api, guest("guest:visitor", ip))

    for response in (live_refusal, ip_refusal):
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "quota_exceeded"
        exposed = response.text + "".join(f"{k}: {v}" for k, v in response.headers.items())
        for secret in (ip, token, sub, "guest:visitor"):
            assert secret not in exposed
