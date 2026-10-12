"""The test network guard: no test, and no Python process a test starts, reaches beyond loopback.

The remote readers are exercised against a local HTTP server only, so no test can touch scroll data, whose
terms each person accepts for themselves before anything reads it.

:func:`install` replaces, in :mod:`socket`, every call that resolves a name or sends to an address -
``getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``, ``gethostbyaddr``, ``getnameinfo``, and the
socket methods ``connect``, ``connect_ex``, ``sendto`` and ``sendmsg`` - with one that raises
:class:`NetworkBlocked` unless the host is loopback. It also removes the environment's proxy variables: a
proxy listening on loopback would pass the guard and then reach the internet on a test's behalf.

``tests/conftest.py`` installs it in the test process and puts ``tests/netguard_site`` first on
``PYTHONPATH``, whose ``sitecustomize`` installs it at the start of every Python subprocess that inherits
the environment. Calls made through the C-level ``_socket`` module directly are not covered.
"""

from __future__ import annotations

import ipaddress
import os
import socket

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


class NetworkBlocked(OSError):
    """Raised when a test tries to reach a non-loopback address."""


def is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host is None or host in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check_host(host, what: str) -> None:
    if not is_loopback(host):
        raise NetworkBlocked(f"tests may not {what} non-loopback host {host!r}")


def _check_address(sock, address, what: str) -> None:
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not is_loopback(address[0]):
        raise NetworkBlocked(f"tests may not {what} {address!r}")


def install() -> None:
    """Install the guard in this process (idempotent)."""
    if getattr(socket, "_apok_netguard", False):
        return
    real = {name: getattr(socket, name) for name in
            ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")}
    cls = socket.socket
    connect, connect_ex, sendto, sendmsg = cls.connect, cls.connect_ex, cls.sendto, cls.sendmsg

    def getaddrinfo(host, *args, **kwargs):
        _check_host(host, "resolve")
        return real["getaddrinfo"](host, *args, **kwargs)

    def gethostbyname(host):
        _check_host(host, "resolve")
        return real["gethostbyname"](host)

    def gethostbyname_ex(host):
        _check_host(host, "resolve")
        return real["gethostbyname_ex"](host)

    def gethostbyaddr(host):
        _check_host(host, "reverse-resolve")
        return real["gethostbyaddr"](host)

    def getnameinfo(sockaddr, flags):
        _check_host(sockaddr[0], "reverse-resolve")
        return real["getnameinfo"](sockaddr, flags)

    def guarded_connect(self, address):
        _check_address(self, address, "connect to")
        return connect(self, address)

    def guarded_connect_ex(self, address):
        _check_address(self, address, "connect to")
        return connect_ex(self, address)

    def guarded_sendto(self, data, *args):
        _check_address(self, args[-1], "send to")  # sendto(data, address) or sendto(data, flags, address)
        return sendto(self, data, *args)

    def guarded_sendmsg(self, buffers, *args):
        if len(args) >= 3 and args[2] is not None:  # sendmsg(buffers, ancdata, flags, address)
            _check_address(self, args[2], "send to")
        return sendmsg(self, buffers, *args)

    socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex = getaddrinfo, gethostbyname, gethostbyname_ex
    socket.gethostbyaddr, socket.getnameinfo = gethostbyaddr, getnameinfo
    cls.connect, cls.connect_ex, cls.sendto, cls.sendmsg = (guarded_connect, guarded_connect_ex, guarded_sendto,
                                                            guarded_sendmsg)
    for name in [n for n in os.environ if n.lower().endswith("_proxy")]:
        del os.environ[name]
    socket._apok_netguard = True
