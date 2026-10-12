"""Object stores: where a volume's bytes come from.

A store maps a *key* to the bytes stored under it, or to ``None`` when no
object exists there. Keys are relative, ``/``-separated, and every segment
matches ``[A-Za-z0-9._-]+`` and is neither ``.`` nor ``..``; anything else is
refused (``IO_STORE_KEY_INVALID``) before any file or network access, so a
key can never climb out of its store or need URL encoding. Keys never contain
``#``, which keeps them valid in an :mod:`apok_eval.merkle` chunk manifest.

:class:`LocalStore` reads files under a root directory, given as a non-empty
path (``""`` would silently mean the working directory) to an existing
directory (``IO_STORE_ROOT_INVALID``). The root is resolved once; every key is
resolved through symlinks and refused if the result leaves the root
(``IO_STORE_ESCAPE``), whether or not the target exists. Symlinks that stay
inside the root are followed. A key that names a directory, a dangling or
looping link or any other non-regular file is refused
(``IO_STORE_NOT_A_FILE``), and so is a key below a dangling or looping link
(a broken link to a directory of chunks must not read as absent chunks). A
key is absent only when it has no entry and its deepest existing ancestor is
a directory, a regular file or a working link to one (as an object store
answers 404 for ``a/b`` when ``a`` is an object). "No entry" means
``ENOENT`` or ``ENOTDIR`` from ``lstat``: any other error on the way (an
unsearchable directory, ``ENAMETOOLONG``, ``EIO``), or in reading the file,
is refused (``IO_STORE_UNREADABLE``, naming the key, the path and the
errno), never read as an absent chunk. The tree is never written to. Reads
are not hardened against concurrent modification of the tree.

:class:`HttpStore` reads objects anonymously with ``urllib``:

* Base URIs: ``s3://<bucket>[/<prefix>]``, mapped by the one canonical rule
  :data:`S3_HTTPS_RULE` (virtual-hosted, region always us-east-1, as villa's
  VC3D does at volume-cartographer/core/src/RemoteUrl.cpp:78-115 @fb8c2c4; its
  other helper drops the region, utils/src/http_fetch.cpp:60-75, and apok-eval does
  not); ``https://`` URIs as given; ``http://`` only to a loopback IP literal,
  ``127.0.0.0/8`` or ``[::1]`` (local mirrors and tests; ``localhost`` is a
  name, which the licence gate never clears). A base URI must already be in
  normal form: printable ASCII without spaces, and unchanged by
  ``urlsplit``/``urlunsplit`` (which lowercase the scheme and drop whitespace,
  control characters and an empty query or fragment), so the URI the
  clearance is asked about is exactly the one fetched. Bucket names must be
  DNS labels without dots (a dotted bucket fails TLS host verification under
  virtual-hosted addressing). Path segments follow the key rule. Anything else
  is ``IO_READ_URI_INVALID``.
* A clearance issued by the licence gate is required: an object whose type
  is exactly :class:`apok_eval.licence.LicenceClearance` (the nominal type, which
  only :func:`apok_eval.licence.check_run` constructs; ``type(c) is``, not
  ``isinstance``, which consults ``__class__`` and so accepts a
  ``Mock(spec=LicenceClearance)``, anything whose ``__class__`` says so, and a
  subclass overriding ``covers``; and not merely an object with a ``covers``
  method, which the runtime-checkable :class:`apok_eval.clearance.Clearance`
  protocol would accept), whose ``covers(base_uri)`` returns ``True`` (the
  bool, not merely a truthy value) at construction; ``covers(object_uri)``
  must return ``True`` before each request, both in the caller's URI form
  (``IO_READ_NOT_CLEARED``, naming the type received or the URI not covered).
  The per-object question is not a formality: the gate maps the gateway path
  ``https://data.aws.ash2txt.org/samples/<k>`` to ``s3://vesuvius-challenge/<k>``,
  so a clearance of that host's root does not cover its ``samples`` objects.
  No request is made for an uncleared URI.
* Requests are plain ``GET`` with a fixed ``User-Agent``: no credentials, no
  ``Authorization`` or cookie header, no signing, no environment proxy, no
  redirect followed. ``http.client`` adds ``Accept-Encoding: identity``, so a
  compliant server sends the stored bytes. TLS is verified.
* Every address the host resolves to is checked immediately before
  connecting, and the connection goes to the checked address (no second
  lookup): unless all are loopback or global, nothing is connected
  (``IO_READ_ADDRESS_FORBIDDEN``). So the instance-metadata service
  (link-local 169.254.169.254, private fd00:ec2::254) is never contacted,
  however the host is spelled (IP literal, a decimal ``2852039166``, or a DNS
  name that resolves there).
* 200 -> the body exactly as served; 404 -> absent; 401 -> refused
  (``IO_READ_AUTH_REQUIRED``: apok-eval sends no credentials); 403 -> refused
  (``IO_READ_FORBIDDEN``: anonymously, S3 answers 403 also for a missing key on
  a non-listable bucket, so a 403 is never read as absent or as fill); 5xx ->
  retried, then refused (``IO_READ_HTTP_STATUS``); any other status, redirects
  included, -> ``IO_READ_HTTP_STATUS`` at once; a connection, TLS, timeout,
  truncated-body or malformed-response error (``http.client``'s ``ValueError``
  for a negative chunk size included) -> retried, then refused
  (``IO_READ_NETWORK``, naming the error and the attempts).
* Retries: a 5xx answer or a transport error is retried up to ``retries``
  times, waiting ``retry_wait_s * attempt`` before retry ``attempt``, so a read
  makes at most ``retries + 1`` requests whatever mix of the two it meets. Each
  retry is appended to ``retry_events`` as ``{"key", "status", "retry"}`` or
  ``{"key", "error", "retry"}`` (``error`` the transport error's class name).
  This holds for ``get``, ``get_base``, ``get_listing``, ``download`` and
  ``get_range``. A refusal raised on the way to a request (an uncleared URI, a
  forbidden address) is never retried. Nor is a local error in ``download``:
  a ``.partial`` file that cannot be opened, written or closed (a missing
  directory, a full disk) is not a failed transfer, so it is refused at once
  (``IO_READ_SINK_UNWRITABLE``, naming the path and the error) and removed, and
  the object is not requested again.
* ``retries`` is an int 0..5, ``retry_wait_s`` 0..3600 and ``timeout_s`` in
  (0, 3600] (``IO_READ_OPTION_INVALID``): beyond that ``time.sleep`` and
  socket timeouts overflow.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.client
import ipaddress
import math
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

from apok_eval import __version__
from apok_eval.fileio import NotRegularFile, read_regular
from apok_eval.licence import LicenceClearance
from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.io.store",
    {
        "IO_STORE_KEY_INVALID": "a store key is not a safe relative key",
        "IO_STORE_ROOT_INVALID": "a local store root is not an existing directory",
        "IO_STORE_ESCAPE": "a store key resolves outside the store root",
        "IO_STORE_NOT_A_FILE": "a store key names something other than a regular file",
        "IO_STORE_UNREADABLE": "a store object, or a directory on the way to it, cannot be read",
        "IO_READ_NOT_CLEARED": "a remote read is not covered by a licence clearance",
        "IO_READ_URI_INVALID": "a remote store URI is malformed or not an allowed form",
        "IO_READ_OPTION_INVALID": "a remote store option is out of range",
        "IO_READ_FORBIDDEN": "a remote object answered HTTP 403",
        "IO_READ_AUTH_REQUIRED": "a remote object answered HTTP 401: the server wants credentials apok-eval does "
                                 "not send",
        "IO_READ_HTTP_STATUS": "a remote object answered an unexpected HTTP status",
        "IO_READ_NETWORK": "a remote read failed at the network, TLS or transfer level",
        "IO_READ_SINK_UNWRITABLE": "a download's local file cannot be written",
        "IO_READ_ADDRESS_FORBIDDEN": "a remote host resolves to an address that is neither loopback nor global",
    },
)

S3_HTTPS_RULE = "s3://<bucket>/<key> -> https://<bucket>.s3.us-east-1.amazonaws.com/<key>"
USER_AGENT = f"apok-eval/{__version__}"
MAX_RETRIES = 5
MAX_WAIT_S = 3600.0  # an hour bounds retry waits and timeouts; 1e300 s overflows time.sleep and socket timeouts

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")
_URI_CHARS = re.compile(r"[\x21-\x7e]+")


@runtime_checkable
class Store(Protocol):
    def get(self, key: str) -> bytes | None:
        """The bytes stored under ``key``, or ``None`` if no object is there."""
        ...

    def describe(self) -> dict:
        """Where this store reads from, as canonical-JSON data."""
        ...


def _safe_segments(segments: list[str]) -> bool:
    return all(s not in (".", "..") and _SEGMENT.match(s) for s in segments)


def check_key(key: str) -> str:
    """Return ``key`` unchanged if it is a safe store key, else refuse."""
    if not isinstance(key, str) or not _safe_segments(key.split("/")):
        raise Refusal("IO_STORE_KEY_INVALID",
                      f"store key {key!r} is not '/'-separated [A-Za-z0-9._-] segments (none empty, '.' or '..')",
                      key=key)
    return key


class LocalStore:
    """Objects as files under ``root``; see the module docstring."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        try:
            resolved = Path(root).resolve(strict=True) if os.fspath(root) else None
        except (OSError, RuntimeError, TypeError, ValueError):  # ValueError: a NUL or an unencodable surrogate
            resolved = None
        if resolved is None or not resolved.is_dir():
            raise Refusal("IO_STORE_ROOT_INVALID", f"local store root {root!r} is not a non-empty path to an "
                          "existing directory", root=str(root))
        self.root = resolved

    def get(self, key: str) -> bytes | None:
        candidate = self.root.joinpath(*check_key(key).split("/"))
        # realpath, not Path.resolve: it treats a symlink loop the same way on every Python >= 3.11.
        resolved = Path(os.path.realpath(candidate))
        if not resolved.is_relative_to(self.root):
            raise Refusal("IO_STORE_ESCAPE", f"key {key!r} resolves to {str(resolved)!r}, outside the store root",
                          key=key, root=str(self.root), resolved=str(resolved))
        try:
            if _has_entry(candidate):
                return read_regular(resolved)
            entry = _deepest_entry(candidate, self.root)
            broken = not _resolves(entry)  # an entry that exists only unresolved: a dangling or looping link
        except OSError as exc:
            if isinstance(exc, NotRegularFile) or exc.errno == errno.ELOOP:
                raise Refusal("IO_STORE_NOT_A_FILE", f"key {key!r} names a directory, a dangling or looping link, "
                              "or another non-regular file", key=key, resolved=str(resolved)) from None
            raise Refusal("IO_STORE_UNREADABLE", f"key {key!r} cannot be read at {exc.filename or resolved}: "
                          f"{exc.strerror or exc}; an unreadable object is never taken as absent", key=key,
                          path=str(exc.filename or resolved), errno=exc.errno) from None
        if broken:
            component = entry.relative_to(self.root).as_posix()
            raise Refusal("IO_STORE_NOT_A_FILE", f"key {key!r} lies below {component!r}, a dangling or looping "
                          "link; what is below a broken link cannot be told absent", key=key,
                          component=component, resolved=str(resolved))
        return None

    def describe(self) -> dict:
        return {"kind": "local", "root": str(self.root)}


def _has_entry(path: Path) -> bool:
    """Whether ``path`` has a directory entry that can be reached: ``lstat`` succeeds. ``ENOENT`` and ``ENOTDIR``
    (a component is missing, or is a file) and ``ELOOP`` (a component above it is a looping link, which
    :func:`_deepest_entry` then finds) mean no reachable entry; any other ``OSError`` propagates."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        return False
    return True


def _resolves(path: Path) -> bool:
    """Whether ``path`` (which has an entry) resolves: ``stat`` follows its links to something. A dangling link
    (``ENOENT``/``ENOTDIR``) or a loop (``ELOOP``) does not; any other ``OSError`` propagates."""
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        return False
    return True


def _deepest_entry(path: Path, root: Path) -> Path:
    """The deepest of ``path`` and its ancestors (up to ``root``) that has a directory entry."""
    while path != root and not _has_entry(path):
        path = path.parent
    return path


def _split(uri: str) -> urllib.parse.SplitResult:
    """``uri`` split into its parts, refused unless it is in normal form (see the module docstring)."""
    try:
        parts = urllib.parse.urlsplit(uri)
        normal = urllib.parse.urlunsplit(parts)
    except ValueError:  # e.g. an unclosed or invalid bracketed IPv6 host
        parts = normal = None
    if normal != uri or not _URI_CHARS.fullmatch(uri):
        raise Refusal("IO_READ_URI_INVALID", f"{uri!r} is not a URI in normal form (printable ASCII without spaces, "
                      f"a lowercase scheme, no empty query or fragment); splitting it gives {normal!r}", uri=uri,
                      normalised=normal)
    return parts


def _checked_path(path: str, uri: str) -> str:
    """``path`` (empty or '/'-led, as urlsplit gives it after a netloc) without one trailing '/',
    after checking its segments by the key rule."""
    trimmed = path[:-1] if path.endswith("/") else path
    if trimmed and not _safe_segments(trimmed[1:].split("/")):
        raise Refusal("IO_READ_URI_INVALID",
                      f"path {path!r} of {uri!r} is not '/'-separated [A-Za-z0-9._-] segments", uri=uri, path=path)
    return trimmed


def s3_to_https(uri: str) -> str:
    """The one HTTPS URL apok-eval reads ``s3://<bucket>[/<key>]`` from (:data:`S3_HTTPS_RULE`)."""
    parts = _split(uri) if isinstance(uri, str) else None
    if parts is None or parts.scheme != "s3" or parts.query or parts.fragment:
        raise Refusal("IO_READ_URI_INVALID", f"{uri!r} is not a plain s3://<bucket>/<key> URI", uri=uri)
    if not _BUCKET.match(parts.netloc):
        raise Refusal("IO_READ_URI_INVALID", f"bucket {parts.netloc!r} is not a 3-63 character DNS label "
                      "of [a-z0-9-] (dots break virtual-hosted TLS)", uri=uri, bucket=parts.netloc)
    return f"https://{parts.netloc}.s3.us-east-1.amazonaws.com{_checked_path(parts.path, uri)}"


def s3_list_url(folder: str) -> str:
    """The anonymous ListObjectsV2 request (``delimiter=/``, at most 1000 keys) for the folder ``s3://<bucket>/<prefix>/``:
    its files and subfolders one level down, from the bucket's endpoint of :data:`S3_HTTPS_RULE`."""
    if not (isinstance(folder, str) and folder.endswith("/")):
        raise Refusal("IO_READ_URI_INVALID", f"{folder!r} is not an s3:// folder ending in '/'", uri=folder)
    url = s3_to_https(folder[:-1])
    endpoint, _, path = url.partition(".amazonaws.com")
    prefix = path.lstrip("/") + "/" if path.strip("/") else ""
    return (f"{endpoint}.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix="
            f"{urllib.parse.quote(prefix, safe='/')}")


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _base_url(uri: str) -> tuple[str, str]:
    """The scheme of ``uri`` and the http(s) URL objects under it are fetched from; refuses forms apok-eval
    does not read."""
    parts = _split(uri) if isinstance(uri, str) else None
    scheme = parts.scheme if parts is not None else None
    if scheme == "s3":
        return scheme, s3_to_https(uri)
    if scheme not in ("https", "http"):
        raise Refusal("IO_READ_URI_INVALID", f"{uri!r}: only s3://, https:// and loopback http:// are read",
                      uri=uri, scheme=scheme)
    try:
        host, _ = parts.hostname, parts.port  # .port raises ValueError when malformed
    except ValueError:
        host = None
    if not host or parts.query or parts.fragment or "@" in parts.netloc:
        raise Refusal("IO_READ_URI_INVALID", f"{uri!r} needs a host and a valid port, and no user info, "
                      "query or fragment", uri=uri)
    if scheme == "http" and not _is_loopback(host):
        raise Refusal("IO_READ_URI_INVALID", f"{uri!r}: plain http is allowed only to a loopback IP literal "
                      "(127.0.0.0/8 or [::1]; the licence gate clears no host name such as localhost)",
                      uri=uri, host=host)
    return scheme, f"{scheme}://{parts.netloc}{_checked_path(parts.path, uri)}"


def _join(base: str, key: str) -> str:
    return (base[:-1] if base.endswith("/") else base) + "/" + key


def _vetted_connection(address: tuple[str, int], *args, **kwargs) -> socket.socket:
    """``socket.create_connection`` to ``address``, refused unless every address its host resolves to is
    loopback or global; the connection is made to a checked numeric address, so nothing resolves twice."""
    host, port = address
    resolved = list(dict.fromkeys(info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    for addr in resolved:
        ip = ipaddress.ip_address(addr)
        if not (ip.is_loopback or ip.is_global):
            raise Refusal("IO_READ_ADDRESS_FORBIDDEN", f"{host!r} resolves to {addr}, which is neither loopback nor "
                          "global (link-local 169.254.169.254 is the instance-metadata service); not connected",
                          host=host, port=port, address=addr)
    for addr in resolved[:-1]:
        try:
            return socket.create_connection((addr, port), *args, **kwargs)
        except OSError:
            continue
    return socket.create_connection((resolved[-1], port), *args, **kwargs)


class _Vetted:
    """Makes an ``http.client`` connection class connect through :func:`_vetted_connection`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _vetted_connection


class _HTTPConnection(_Vetted, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_Vetted, http.client.HTTPSConnection):
    pass


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(_HTTPSConnection, req, context=self._context)


class _Sink:
    """A download's ``.partial`` file. An ``OSError`` opening, writing or closing it is kept (``error``, the first
    one), not raised: it is a local error, not a failed transfer, so ``HttpStore._fetch`` refuses it at once
    instead of retrying the request."""

    def __init__(self, path: Path) -> None:
        self.error: OSError | None = None
        try:
            self._file = open(path, "wb")
        except OSError as exc:
            self.error, self._file = exc, None

    def write(self, block: bytes) -> None:
        try:
            self._file.write(block)
        except OSError as exc:
            self.error = exc

    def close(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
        except OSError as exc:
            self.error = self.error or exc


def _finite_at_least(value: object, low: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value >= low)


class HttpStore:
    """Anonymous objects under a cleared base URI; see the module docstring."""

    def __init__(self, base_uri: str, clearance: LicenceClearance, *, retries: int = 0, retry_wait_s: float = 1.0,
                 timeout_s: float = 60.0) -> None:
        scheme, self.base_url = _base_url(base_uri)
        received = f"{type(clearance).__module__}.{type(clearance).__qualname__}"
        # type(), not isinstance(): isinstance consults __class__, which a Mock(spec=LicenceClearance) or any
        # object can claim, and it accepts a subclass overriding covers(); only the gate's own type will do.
        if type(clearance) is not LicenceClearance:
            raise Refusal("IO_READ_NOT_CLEARED", f"a {received} is not a clearance issued by the licence gate "
                          "(exactly apok_eval.licence.LicenceClearance, from apok_eval.licence.check_run; not a subclass, a "
                          "mock with its spec, or anything else with a covers() method). Nothing under "
                          f"{base_uri!r} will be requested", uri=base_uri, clearance_type=received)
        if clearance.covers(base_uri) is not True:
            raise Refusal("IO_READ_NOT_CLEARED", f"the licence clearance does not cover {base_uri!r}; nothing will "
                          "be requested", uri=base_uri, clearance_type=received)
        options = (
            ("retries", retries,
             isinstance(retries, int) and not isinstance(retries, bool) and 0 <= retries <= MAX_RETRIES),
            ("retry_wait_s", retry_wait_s, _finite_at_least(retry_wait_s, 0) and retry_wait_s <= MAX_WAIT_S),
            ("timeout_s", timeout_s, _finite_at_least(timeout_s, 0) and 0 < timeout_s <= MAX_WAIT_S),
        )
        for name, value, ok in options:
            if not ok:
                raise Refusal("IO_READ_OPTION_INVALID", f"{name}={value!r} is out of range (retries an int "
                              f"0..{MAX_RETRIES}; waits 0..{MAX_WAIT_S:g} s; timeout above 0, at most "
                              f"{MAX_WAIT_S:g} s)", field=name, value=value)
        self.base_uri = base_uri
        self.s3_mapping = S3_HTTPS_RULE if scheme == "s3" else None
        self.retries, self.retry_wait_s, self.timeout_s = retries, float(retry_wait_s), float(timeout_s)
        self.retry_events: list[dict] = []
        self._clearance = clearance
        # A bare director: no proxy, redirect, auth, cookie or error-raising handler, so every
        # status reaches get() exactly as served; its handlers connect only to vetted addresses.
        self._opener = urllib.request.OpenerDirector()
        self._opener.add_handler(_HTTPHandler())
        self._opener.add_handler(_HTTPSHandler(context=ssl.create_default_context()))

    def object_uri(self, key: str) -> str:
        """``key`` under the base URI, in the caller's form (what the clearance is asked about)."""
        return _join(self.base_uri, check_key(key))

    def get(self, key: str) -> bytes | None:
        self._require_cleared(self.object_uri(key), key)
        return self._get(_join(self.base_url, key), key)

    def _require_cleared(self, uri: str, key: str) -> None:
        """Every read asks the gate about its own URI first, and only the bool True clears it."""
        if self._clearance.covers(uri) is not True:
            raise Refusal("IO_READ_NOT_CLEARED", f"the licence clearance does not cover {uri!r}; not requested",
                          uri=uri, key=key)

    def get_base(self) -> bytes | None:
        """The object at the base URI itself, for a base that names one declared file (e.g. a JSON file)."""
        self._require_cleared(self.base_uri, "")
        return self._get(self.base_url, "")

    def get_listing(self, key: str = "") -> bytes | None:
        """The server's listing of the folder ``key`` (the base itself when empty). Over http(s) it is the index
        page the folder's URL, ending in '/', serves; under an ``s3://`` base it is the bucket's anonymous
        ListObjectsV2 answer for the folder's prefix, one level deep (:func:`s3_list_url`). Either way the folder's
        URI, ending in '/', is what the clearance must cover."""
        folder = (self.object_uri(key) if key else self.base_uri.rstrip("/")) + "/"
        self._require_cleared(folder, key)
        if self.s3_mapping is not None:
            return self._get(s3_list_url(folder), key + "/")
        return self._get((_join(self.base_url, key) if key else self.base_url.rstrip("/")) + "/", key + "/")

    def download(self, key: str, path: str | os.PathLike[str], *, chunk: int = 8 << 20) -> tuple[int, str] | None:
        """Stream ``key`` to ``path`` without holding it in memory: ``(size, "sha256:…")``, or None when absent.

        The same statuses, retries and refusals as ``get``. The file is written under a temporary name and moved
        into place only when complete; the temporary file is removed after every failed attempt, so a retry starts
        afresh and a failed or refused read leaves nothing at ``path``. A temporary file that cannot be opened, written
        or closed is a local error, not a failed transfer: refused at once (``IO_READ_SINK_UNWRITABLE``), never retried.
        """
        self._require_cleared(self.object_uri(key), key)
        url = _join(self.base_url, key)
        target = Path(path)
        partial = target.with_name(target.name + ".partial")
        attempt = 0
        while True:
            attempt += 1
            status, got = self._fetch(url, key, sink=partial, chunk=chunk)
            if status == 200:
                os.replace(partial, target)
                return got
            partial.unlink(missing_ok=True)
            if status == 404:
                return None
            self._retry_or_refuse(url, key, status, attempt, got)

    def _retry_or_refuse(self, url: str, key: str, status: int | None, attempt: int, cause: object = None) -> None:
        """After an answer other than 200 or 404, or a failed transfer (``status`` None, ``cause`` the error): wait
        and return for a retry, or refuse."""
        if status is None:
            error = type(cause).__name__
            if attempt <= self.retries:
                self._wait(key, {"error": error}, attempt)
                return
            raise Refusal("IO_READ_NETWORK", f"{url}: {error}: {cause}, after {attempt} attempt(s)", key=key,
                          url=url, error=error, attempts=attempt)
        if 500 <= status <= 599 and attempt <= self.retries:
            self._wait(key, {"status": status}, attempt)
            return
        if status == 401:  # reads are anonymous: credentials are never sent from here
            raise Refusal("IO_READ_AUTH_REQUIRED", f"{url} answered 401: it wants credentials, and apok-eval reads "
                          "anonymously", key=key, url=url, status=status)
        if status == 403:
            raise Refusal("IO_READ_FORBIDDEN", f"{url} answered 403; anonymously that cannot be told from a "
                          "missing key on a non-listable bucket, so it is neither absent nor fill",
                          key=key, url=url, status=status)
        raise Refusal("IO_READ_HTTP_STATUS", f"{url} answered HTTP {status} after {attempt} attempt(s)",
                      key=key, url=url, status=status, attempts=attempt)

    def _wait(self, key: str, failure: dict, attempt: int) -> None:
        """Record retry ``attempt`` of ``key`` and what it follows, then wait ``retry_wait_s * attempt``."""
        self.retry_events.append({"key": key, **failure, "retry": attempt})
        time.sleep(self.retry_wait_s * attempt)

    def _get(self, url: str, key: str) -> bytes | None:
        attempt = 0
        while True:
            attempt += 1
            status, body = self._fetch(url, key)
            if status == 200:
                return body
            if status == 404:
                return None
            self._retry_or_refuse(url, key, status, attempt, body)

    def get_range(self, key: str, start: int, length: int) -> bytes | None:
        """``length`` bytes of ``key`` from byte ``start`` (fewer at its end), cleared like ``get``: the server's 206
        answer, or the slice of a 200 answer from a server that ignores ranges. None when absent."""
        if not (isinstance(start, int) and isinstance(length, int) and start >= 0 and 0 < length <= 1 << 26):
            raise Refusal("IO_READ_OPTION_INVALID", f"byte range start={start!r} length={length!r} is out of range "
                          "(start >= 0, 0 < length <= 64 MiB)", field="range", value=[start, length])
        self._require_cleared(self.object_uri(key), key)
        url = _join(self.base_url, key)
        headers = {"Range": f"bytes={start}-{start + length - 1}"}
        attempt = 0
        while True:
            attempt += 1
            status, body = self._fetch(url, key, headers=headers)
            if status == 206:
                return bytes(body)[:length]
            if status == 200:
                return bytes(body)[start:start + length]
            if status == 404:
                return None
            if status == 416:  # the range starts past the end
                return b""
            self._retry_or_refuse(url, key, status, attempt, body)

    def _fetch(self, url: str, key: str, sink: Path | None = None, chunk: int = 8 << 20,
               headers: dict[str, str] | None = None) -> tuple[int | None, object]:
        """One request: ``(status, body)``, ``(200, (size, digest))`` when streamed to ``sink``, or ``(None, error)``
        when the transfer failed (the caller retries or refuses). A refusal raised while connecting propagates, and so
        does ``IO_READ_SINK_UNWRITABLE`` when ``sink`` cannot be opened, written or closed (the sink is removed)."""
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                if sink is None or response.status != 200:
                    return response.status, response.read()
                digest, size, out = hashlib.sha256(), 0, _Sink(sink)
                try:
                    while out.error is None and (block := response.read(chunk)):
                        out.write(block)
                        digest.update(block)
                        size += len(block)
                finally:
                    out.close()
                if out.error is not None:  # a local error: not a failed transfer, so never retried or re-requested
                    with contextlib.suppress(OSError):
                        sink.unlink()
                    error = type(out.error).__name__
                    raise Refusal("IO_READ_SINK_UNWRITABLE", f"{sink}: {error}: {out.error}; the download of {key!r} "
                                  "cannot be written here, and a local error is not retried", key=key, path=str(sink),
                                  error=error)
                declared = response.getheader("Content-Length")
                if declared is not None and declared.isdigit() and int(declared) != size:
                    # read(n) returns what arrived when the connection closes early; read() would have raised
                    raise http.client.IncompleteRead(b"", int(declared) - size)
                return response.status, (size, "sha256:" + digest.hexdigest())
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # Name what failed, not urllib's wrapper: with a host and an http(s) handler always present,
            # urllib raises URLError only around the underlying OSError. http.client raises ValueError for
            # some malformed responses (a negative chunk size: "read length must be non-negative").
            # A Refusal (IO_READ_ADDRESS_FORBIDDEN from _vetted_connection) is none of these, so it propagates.
            return None, exc.reason if isinstance(exc, urllib.error.URLError) else exc

    def describe(self) -> dict:
        return {"kind": "http", "base_uri": self.base_uri, "base_url": self.base_url,
                "s3_mapping": self.s3_mapping,
                "retries": self.retries, "retry_events": [dict(e) for e in self.retry_events]}
