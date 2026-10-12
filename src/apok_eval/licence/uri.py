"""One canonical form for every location the licence gate compares.

Registry hosts, asset locations, run-configuration inputs and the URIs a reader
later asks a clearance about are all mapped through :func:`canonical_uri`
first, so two spellings of one object can never get two different answers.

Canonical forms (``str(CanonicalURI)``; never a trailing slash, the root of a
bucket or host is written without one; the one exception is the local
filesystem root, ``file:///``):

* ``s3://<bucket>/<key>`` for every spelling of an S3 object the gate
  recognises: ``s3://b/k``; virtual-hosted ``https://b.s3.amazonaws.com/k``
  and ``https://b.s3.<region>.amazonaws.com/k``; path-style
  ``https://s3[.<region>].amazonaws.com/b/k``; and the Vesuvius Challenge
  gateway ``https://data.aws.ash2txt.org/samples/k``, which serves
  ``s3://vesuvius-challenge/k`` (villa fb8c2c4
  ``scrollprize.org/src/data/atlasDataAccess.json:8-15``,
  ``volume-cartographer/apps/VC3D/OpenDataManifest.cpp:774-789``).
  Both https forms of the open-data bucket (with and without
  ``us-east-1``) therefore become ``s3://vesuvius-challenge-open-data/k``.
* ``https://<host>/<path>`` for any other https URL (host lower-cased),
  e.g. ``https://dl.ash2txt.org/...``.
* ``http://<loopback ip>[:<port>]/<path>``: plain http is accepted only for a
  loopback literal (127.0.0.0/8 or ``[::1]``), so readers can be tested against
  a local server; ``:80`` is dropped.
* ``file:///<path>`` for an absolute local path or a ``file:///`` URI, after
  ``os.path.realpath``: symlinks are resolved, so a link inside a declared
  root that points outside it is judged by where it really points.
* ``synthetic://<name>/<path>`` names data apok-eval generates itself.

An https spelling that would name an object the gate cannot map to one form
is refused rather than kept as a plain https host, because a plain host is
judged only by the registry entries that name it: any other host under
``amazonaws.com`` (dualstack, FIPS, accelerate, ``s3-<region>``, website and
``s3-external-1`` endpoints, or ``s3.amazonaws.com`` without a bucket), the
Vesuvius Challenge zarr proxy ``zarr.aws.ash2txt.org``, which re-serves both
buckets under its own paths (``atlasDataAccess.json:33-44``), and an IP
address, which may be any host.

Path handling: ``.`` segments are dropped and ``..`` removes the previous
segment (before the S3 gateway mapping, as an HTTP client would); ``..`` above
the bucket or host root is refused, as is an empty segment (``a//b``), because
S3 keys may contain ``//`` literally.

Refused, never guessed (each refusal names ``what`` - the field being read -
and the offending ``uri``):

* ``LICENCE_URI_MALFORMED`` - not a non-empty string; whitespace, control,
  invisible or backslash characters; invalid bucket or host name; a bad port;
  a ``file:`` URI that is not ``file:///<absolute path>``; a local path whose
  real location contains a character refused here.
* ``LICENCE_URI_AMBIGUOUS`` - percent-encoding (``s3://`` keys are literal but
  https paths are decoded, so ``%2F`` would mean two things), a query or
  fragment, userinfo, an explicit port on a non-http scheme, an empty path
  segment, ``..`` above the root, or an https host that may be another
  spelling of a known object (an unrecognised ``amazonaws.com`` endpoint, the
  zarr proxy, an IP address; ``host`` names it).
* ``LICENCE_URI_RELATIVE`` - no scheme and not an absolute path, a
  scheme-relative ``//host/...``, or a scheme with no host or bucket.
* ``LICENCE_URI_SCHEME`` - any scheme but s3, https, file, synthetic, and
  http to anything but a loopback literal (``zarr2://``, ``s3+region://`` and
  ``localhost`` included: the latter resolves by name).

Prefix tests (:meth:`CanonicalURI.is_under`) compare whole path segments, so
``.../PHerc0332`` covers ``.../PHerc0332/x`` but not ``.../PHerc03320``.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass

from apok_eval.refusal import Refusal, register_codes

register_codes(
    "apok_eval.licence.uri",
    {
        "LICENCE_URI_MALFORMED": "a URI or path is not a usable string, or has an invalid bucket, host or port",
        "LICENCE_URI_AMBIGUOUS": "a URI could name more than one object (percent-encoding, query, fragment, "
                                 "userinfo, port, empty segment, '..' above its root, or an unrecognised S3 "
                                 "endpoint, proxy or IP host)",
        "LICENCE_URI_RELATIVE": "a URI does not name its scheme and host or bucket",
        "LICENCE_URI_SCHEME": "a URI uses a scheme the gate does not canonicalise (http only for loopback)",
    },
)

_SCHEMES = ("s3", "https", "http", "file", "synthetic")
#: Schemes whose objects live on this machine (http is accepted for loopback only).
LOCAL_SCHEMES = ("file", "synthetic", "http")

_SCHEME_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*):")
_LABEL = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
_HOST_RE = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_REGION = r"[a-z]{2}(?:-[a-z]+)+-[0-9]+"
_S3_VIRTUAL_HOST = re.compile(rf"(?P<bucket>[a-z0-9.-]+)\.s3(?:\.{_REGION})?\.amazonaws\.com")
_S3_PATH_HOST = re.compile(rf"s3(?:\.{_REGION})?\.amazonaws\.com")
_HOST_PORT = re.compile(r"(?:\[(?P<v6>[^\]]+)\]|(?P<v4>[^:\[\]]+))(?::(?P<port>[0-9]+))?")
_PORT_RE = re.compile(r"[1-9][0-9]{0,4}")

# The Vesuvius Challenge full-dataset gateway serves s3://vesuvius-challenge under /samples/
# (villa fb8c2c4 scrollprize.org/src/data/atlasDataAccess.json:12-15).
_GATEWAY_HOST, _GATEWAY_ROOT, _GATEWAY_BUCKET = "data.aws.ash2txt.org", "samples", "vesuvius-challenge"
_AWS_DOMAIN = "amazonaws.com"
# Re-serves both buckets under /data and /open-data (villa fb8c2c4 scrollprize.org/src/data/atlasDataAccess.json:33-44).
_ZARR_PROXY_HOST = "zarr.aws.ash2txt.org"


@dataclass(frozen=True)
class CanonicalURI:
    """A location in canonical form; ``str()`` gives the canonical spelling."""

    scheme: str
    authority: str
    segments: tuple[str, ...]

    def __str__(self) -> str:
        if self.scheme == "file" and not self.segments:
            return "file:///"  # "file://" names no path and would not canonicalise again
        return f"{self.scheme}://{self.authority}" + "".join("/" + s for s in self.segments)

    @property
    def is_local(self) -> bool:
        return self.scheme in LOCAL_SCHEMES

    def is_under(self, prefix: CanonicalURI) -> bool:
        """True if ``prefix`` equals this location or is a whole-segment prefix of it."""
        same_root = (self.scheme, self.authority) == (prefix.scheme, prefix.authority)
        return same_root and self.segments[:len(prefix.segments)] == prefix.segments


def _forbidden_char(text: str) -> str | None:
    for ch in text:
        if ch == "\\" or ch.isspace() or not ch.isprintable():
            return ch
    return None


def _marker(text: str) -> str | None:
    return next((ch for ch in "%?#" if ch in text), None)


def canonical_uri(uri: object, what: str = "uri") -> CanonicalURI:
    """Canonicalise ``uri`` (see the module docstring) or refuse naming ``what``."""
    if not isinstance(uri, str) or not uri:
        raise Refusal("LICENCE_URI_MALFORMED", f"{what} must be a non-empty string", what=what, uri=uri)
    bad = _forbidden_char(uri)
    if bad is not None:
        raise Refusal("LICENCE_URI_MALFORMED", f"{what} contains the forbidden character {bad!r}",
                      what=what, uri=uri, char=bad)
    marker = _marker(uri)
    if marker is not None:
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what} contains {marker!r} (percent-encoding, query or fragment); "
                      "write the plain object location", what=what, uri=uri, char=marker)
    if uri.startswith("//"):
        raise Refusal("LICENCE_URI_RELATIVE", f"{what} is scheme-relative ('//...'); write the scheme",
                      what=what, uri=uri)
    if uri.startswith("/"):
        return _local(uri, what, uri)
    match = _SCHEME_RE.match(uri)
    if match is None:
        raise Refusal("LICENCE_URI_RELATIVE", f"{what} is a relative path or has no scheme; write an absolute "
                      "path or a full URI", what=what, uri=uri)
    scheme, rest = match.group(1).lower(), uri[match.end():]
    if scheme not in _SCHEMES:
        raise Refusal("LICENCE_URI_SCHEME", f"{what} uses scheme {scheme!r}; the gate reads only "
                      f"{', '.join(_SCHEMES)}", what=what, uri=uri, scheme=scheme)
    if not rest.startswith("//"):
        raise Refusal("LICENCE_URI_RELATIVE", f"{what}: '{scheme}:' is not followed by '//' and a host",
                      what=what, uri=uri)
    authority, slash, tail = rest[2:].partition("/")
    path = slash + tail
    if scheme == "file":
        if authority or not path:
            raise Refusal("LICENCE_URI_MALFORMED", f"{what}: a file URI must be file:///<absolute path>",
                          what=what, uri=uri)
        return _local(path, what, uri)
    return _remote(scheme, authority, path, what, uri)


def _remote(scheme: str, authority: str, path: str, what: str, uri: str) -> CanonicalURI:
    if not authority:
        raise Refusal("LICENCE_URI_RELATIVE", f"{what} names no host or bucket", what=what, uri=uri)
    if "@" in authority:
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what} carries userinfo ('@') in its authority",
                      what=what, uri=uri)
    segments = _resolve(path, what, uri)
    if scheme == "http":
        return CanonicalURI("http", _loopback(authority, what, uri), segments)
    if ":" in authority:
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what} carries an explicit port", what=what, uri=uri)
    if scheme == "s3":
        return CanonicalURI("s3", _bucket(authority, what, uri), segments)
    host = authority.lower()
    if not _HOST_RE.fullmatch(host):
        raise Refusal("LICENCE_URI_MALFORMED", f"{what}: {authority!r} is not a valid host name",
                      what=what, uri=uri, host=authority)
    if scheme == "synthetic":
        return CanonicalURI("synthetic", host, segments)
    return _https(host, segments, what, uri)


def _resolve(path: str, what: str, uri: str) -> tuple[str, ...]:
    if not path:
        return ()
    parts = path.split("/")[1:]
    if parts[-1] == "":
        parts.pop()  # one trailing slash names the same prefix
    out: list[str] = []
    for part in parts:
        if part == "":
            raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what} has an empty path segment ('//')", what=what, uri=uri)
        if part == "..":
            if not out:
                raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what} climbs above its root with '..'",
                              what=what, uri=uri)
            out.pop()
        elif part != ".":
            out.append(part)
    return tuple(out)


def _bucket(name: str, what: str, uri: str) -> str:
    if not _BUCKET_RE.fullmatch(name) or ".." in name:
        raise Refusal("LICENCE_URI_MALFORMED", f"{what}: {name!r} is not a valid S3 bucket name",
                      what=what, uri=uri, bucket=name)
    return name


def _https(host: str, segments: tuple[str, ...], what: str, uri: str) -> CanonicalURI:
    if host.rpartition(".")[2].isdigit():
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what}: {host!r} is an IP address, which may serve any host; "
                      "write the host name", what=what, uri=uri, host=host)
    virtual = _S3_VIRTUAL_HOST.fullmatch(host)
    if virtual is not None:
        return CanonicalURI("s3", _bucket(virtual.group("bucket"), what, uri), segments)
    if _S3_PATH_HOST.fullmatch(host) and segments:
        return CanonicalURI("s3", _bucket(segments[0], what, uri), segments[1:])
    if host == _GATEWAY_HOST and segments[:1] == (_GATEWAY_ROOT,):
        return CanonicalURI("s3", _GATEWAY_BUCKET, segments[1:])
    if host == _AWS_DOMAIN or host.endswith("." + _AWS_DOMAIN):
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what}: {host!r} is not an S3 endpoint the gate maps to "
                      "s3://<bucket>/<key>, so its objects could carry a second name; write the s3:// URI",
                      what=what, uri=uri, host=host)
    if host == _ZARR_PROXY_HOST:
        raise Refusal("LICENCE_URI_AMBIGUOUS", f"{what}: {host!r} is a proxy that re-serves the Vesuvius Challenge "
                      "buckets under its own paths; write the bucket's s3:// URI", what=what, uri=uri, host=host)
    return CanonicalURI("https", host, segments)


def _loopback(authority: str, what: str, uri: str) -> str:
    match = _HOST_PORT.fullmatch(authority)
    try:
        ip = ipaddress.ip_address(match.group("v6") or match.group("v4")) if match else None
    except ValueError:
        ip = None
    if ip is None or not ip.is_loopback:
        raise Refusal("LICENCE_URI_SCHEME", f"{what}: plain http is accepted only for a loopback test server "
                      "(127.0.0.0/8 or [::1])", what=what, uri=uri, host=authority)
    port = match.group("port")
    if port is not None and (not _PORT_RE.fullmatch(port) or int(port) > 65535):
        raise Refusal("LICENCE_URI_MALFORMED", f"{what}: {port!r} is not a valid port", what=what, uri=uri,
                      port=port)
    host = f"[{ip}]" if ip.version == 6 else str(ip)
    return host if port in (None, "80") else f"{host}:{port}"


def _local(path: str, what: str, uri: str) -> CanonicalURI:
    real = os.path.realpath(path)
    bad = _forbidden_char(real) or _marker(real)
    if bad is not None:
        raise Refusal("LICENCE_URI_MALFORMED", f"{what} resolves to {real!r}, which contains {bad!r}",
                      what=what, uri=uri, real=real, char=bad)
    return CanonicalURI("file", "", tuple(part for part in real.split("/") if part))
