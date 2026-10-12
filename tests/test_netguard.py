"""The test network guard (tests/netguard.py) blocks every way out, in this process and in subprocesses."""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest
from netguard import NetworkBlocked

OUTSIDE = "192.0.2.1"  # TEST-NET-1 (RFC 5737): never routed, and still not loopback


@pytest.mark.parametrize("call", [
    lambda: socket.getaddrinfo("example.org", 443),
    lambda: socket.gethostbyname("example.org"),
    lambda: socket.gethostbyname_ex("example.org"),
    lambda: socket.gethostbyaddr(OUTSIDE),
    lambda: socket.getnameinfo((OUTSIDE, 80), 0),
    lambda: socket.create_connection((OUTSIDE, 80), timeout=1),
])
def test_resolving_or_connecting_beyond_loopback_is_blocked(call):
    with pytest.raises(NetworkBlocked):
        call()


@pytest.mark.parametrize("send", [
    lambda s: s.sendto(b"x", (OUTSIDE, 9)),
    lambda s: s.sendto(b"x", 0, (OUTSIDE, 9)),
    lambda s: s.sendmsg([b"x"], [], 0, (OUTSIDE, 9)),
    lambda s: s.connect((OUTSIDE, 9)),
    lambda s: s.connect_ex((OUTSIDE, 9)),
])
def test_udp_datagrams_beyond_loopback_are_blocked(send):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, pytest.raises(NetworkBlocked):
        send(sock)


def test_loopback_still_works():
    assert socket.gethostbyname("localhost").startswith("127.")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        assert sender.sendto(b"ping", receiver.getsockname()) == 4
        assert sender.sendmsg([b"pong"], [], 0, receiver.getsockname()) == 4
        assert receiver.recv(16) == b"ping" and receiver.recv(16) == b"pong"


def test_a_python_subprocess_is_guarded_too():
    code = ("import socket\n"
            "for call in (lambda: socket.gethostbyname('example.org'),\n"
            "             lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'x', ('192.0.2.1', 9))):\n"
            "    try:\n"
            "        call()\n"
            "    except OSError as exc:\n"
            "        print(type(exc).__name__)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True).stdout
    assert out.split() == ["NetworkBlocked", "NetworkBlocked"]
