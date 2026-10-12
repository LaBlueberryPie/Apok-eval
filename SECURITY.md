# Security policy

apok-eval is maintained by Static Signal Ltd. We take reports of vulnerabilities seriously and are grateful
to anyone who reports one responsibly.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1.0 | No |

Security fixes are made on the latest release line.

## Reporting a vulnerability

**Please do not report vulnerabilities in public issues, pull requests or discussions.** Email
[hello@staticsignal.co.uk](mailto:hello@staticsignal.co.uk) instead, with "apok-eval security" in the subject.

Please include:

- the apok-eval version (`apok-eval --version`) or the commit you tested, your Python version and your
  operating system;
- the component affected (see [Scope](#scope)) and the command or function involved;
- a description of the problem and its impact: what an attacker controls, and what they gain;
- the smallest reproduction you can give, built on **synthetic** inputs: a crafted file, a small script, or
  the steps that produce one. Do not send scroll data, results computed on scroll data, credentials or
  personal data;
- whether the problem is known to anyone else, and any disclosure date you are working to.

## What happens next

- We aim to acknowledge your report within a few working days.
- We confirm or rule out the problem, and keep you informed of our progress.
- Once a fix is released, we will normally publish a GitHub security advisory that describes the problem and
  the fixed version, crediting you there if you wish.

We ask that you give us reasonable time to release a fix before disclosing the problem publicly. There is
no bug bounty: we cannot offer payment for reports.

## Scope

apok-eval reads files and volumes that other people prepare, and its records are meant to be checked by
people who did not write them. Reports in these areas are in scope:

- **The licence gate** (`apok_eval.licence`): any way to read an input, locally or remotely, that the gate
  has not cleared when it is in use; to obtain a clearance that covers a URI the gate did not clear; or to
  make the gate accept a licence, host or acceptance of terms that the registry and the run configuration do
  not support ([docs/licence.md](docs/licence.md)).
- **The ledger's integrity checks** (`apok_eval.ledger`): any way to make `apok-eval verify` pass a record
  whose inputs, figures, code identity or seal were changed, beyond what
  [docs/ledger.md](docs/ledger.md) states that `verify` cannot establish.
- **The readers' refusal of malformed or hostile inputs** (`apok_eval.io`, `apok_eval.eval.fibres`,
  `apok_eval.jsonstrict`): a file or store that makes a reader fail with anything other than a refusal,
  exhaust memory or CPU, or hang; or a key that makes a local store read outside its root (`IO_STORE_ESCAPE`).
  A symlink that the documentation says is followed and recorded, such as a tifxyz core file, is not an escape.
- **Network access rules** (`apok_eval.io.store`): any way to make apok-eval contact a host other than the
  one cleared, follow a redirect, send credentials, use plain `http://` to anything but a loopback IP
  literal, or reach a link-local or private address such as an instance-metadata service.

Out of scope:

- the limits the documentation already states (for example, "What `verify` cannot establish" in
  [docs/ledger.md](docs/ledger.md) and "What is enforced, what is policy, what is yours" in
  [docs/licence.md](docs/licence.md)), although reports that a stated limit is wrong are welcome;
- deliberate forgery by code running in the same Python process, which Python cannot prevent ("The
  clearance" in [docs/licence.md](docs/licence.md));
- vulnerabilities in dependencies such as numpy, scipy, tifffile, numcodecs or imagecodecs, which should be
  reported to those projects;
- the test suite's network guard (`tests/netguard.py`), which keeps tests on loopback and is not a security
  boundary;
- the interpretation of data licences and terms, which is for the data's owners and for you.
