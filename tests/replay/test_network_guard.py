"""Trace: docs/test-plan.md section 13, row "Zero-network replay" (any socket connect raises).

A guard that cannot fail proves nothing, so these tests show it fires on every connect path
and show that the same connect succeeds without the guard.
"""

from __future__ import annotations

import socket

import pytest

from tests.replay.support import NetworkAccessError

pytestmark = pytest.mark.replay


@pytest.fixture
def loopback_server():
    """A real listening socket, so a refused connect is the guard's doing and nothing else."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        yield server.getsockname()
    finally:
        server.close()


def test_connect_raises_while_the_same_connect_works_without_the_guard(
    network_guard, loopback_server
):
    """Zero-network replay: connect to a reachable local port raises; the unguarded connect succeeds."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAccessError):
            client.connect(loopback_server)
    finally:
        client.close()
    network_guard.attempts.clear()

    control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        network_guard.real_connect(control, loopback_server)
        assert control.getpeername() == loopback_server
    finally:
        control.close()


def test_connect_ex_raises(network_guard, loopback_server):
    """Zero-network replay: connect_ex is covered, not just connect."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAccessError):
            client.connect_ex(loopback_server)
    finally:
        client.close()
    network_guard.attempts.clear()


def test_create_connection_raises(network_guard, loopback_server):
    """Zero-network replay: socket.create_connection (used by http clients) is covered."""
    with pytest.raises(NetworkAccessError):
        socket.create_connection(loopback_server, timeout=1)
    network_guard.attempts.clear()


def test_ipv6_connect_raises(network_guard):
    """Zero-network replay: the guard is not limited to IPv4."""
    try:
        client = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("IPv6 sockets are unavailable on this host")
    try:
        with pytest.raises(NetworkAccessError):
            client.connect(("::1", 9, 0, 0))
    finally:
        client.close()
    network_guard.attempts.clear()


async def test_unix_sockets_and_the_event_loop_still_work():
    """The guard must not break asyncio internals, which use AF_UNIX socket pairs."""
    import asyncio

    left, right = socket.socketpair()
    try:
        loop = asyncio.get_running_loop()
        await loop.sock_sendall(left, b"ping")
        left.setblocking(False)
        right.setblocking(False)
        assert await loop.sock_recv(right, 4) == b"ping"
        assert await asyncio.to_thread(lambda: 21 * 2) == 42
    finally:
        left.close()
        right.close()


def test_unrecorded_attempts_fail_the_test_at_teardown(network_guard):
    """A swallowed NetworkAccessError must still be visible: attempts are recorded."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            client.connect(("127.0.0.1", 9))
        except NetworkAccessError:
            pass  # what a production `except Exception` would do
    finally:
        client.close()
    assert network_guard.attempts == ["('127.0.0.1', 9)"]
    network_guard.attempts.clear()  # acknowledged here; any other test would fail at teardown
