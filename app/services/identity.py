"""API identity (ADR-0010): the development API key or a JWT from an allow-listed issuer."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import jwt

from app.config import JwtIssuer

ROLES = ("guest", "analyst", "admin")


class UnauthorizedError(Exception):
    """The credential is missing, malformed, unverifiable or carries unacceptable claims."""


@dataclass(frozen=True)
class Principal:
    customer_id: str
    role: str
    ip_hash: str | None = None


def authenticate(
    *,
    bearer: str | None,
    x_api_key: str | None,
    api_key: str | None,
    issuers: list[JwtIssuer],
) -> Principal:
    supplied = x_api_key or bearer
    if not supplied:
        raise UnauthorizedError
    if api_key and secrets.compare_digest(supplied.encode(), api_key.encode()):
        return Principal(customer_id=hashlib.sha256(supplied.encode()).hexdigest(), role="admin")
    if x_api_key:
        raise UnauthorizedError
    return _verify_token(supplied, issuers)


def _verify_token(token: str, issuers: list[JwtIssuer]) -> Principal:
    try:
        # The unverified `iss` only selects which key to verify with; it is never trusted.
        claimed_issuer = jwt.decode(token, options={"verify_signature": False}).get("iss")
        issuer = next((entry for entry in issuers if entry.issuer == claimed_issuer), None)
        if issuer is None:
            raise UnauthorizedError
        claims = jwt.decode(
            token,
            issuer.public_key_pem,
            algorithms=["ES256"],
            audience=issuer.audience,
            issuer=issuer.issuer,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise UnauthorizedError from exc
    subject, role = claims["sub"], claims.get("role")
    if (
        not isinstance(subject, str)
        or not subject
        or not isinstance(role, str)
        or role not in ROLES
    ):
        raise UnauthorizedError
    # The guest quota key: the salted IP hash the BFF signs in (ADR-0010 amendment).
    ip_hash = claims.get("ip_hash") if role == "guest" else None
    if role == "guest" and (not isinstance(ip_hash, str) or not ip_hash):
        raise UnauthorizedError
    return Principal(customer_id=subject, role=role, ip_hash=ip_hash)
