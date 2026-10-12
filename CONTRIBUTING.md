# Contributing to apok-eval

Thank you for your interest in apok-eval. Bug reports, fixes, documentation improvements and well-argued
proposals are all welcome. apok-eval is maintained by Static Signal Ltd.

- Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Security vulnerabilities are reported privately, as [SECURITY.md](SECURITY.md) describes, never in an
  issue or a pull request.
- Bugs and proposals go to the [issue tracker](https://github.com/LaBlueberryPie/apok-eval/issues).

## Before you start

- **Discuss larger changes first.** For anything beyond a small fix, open an issue to agree the approach
  before you write the code. apok-eval ships the evaluator, the readers it needs, the run ledger and the
  licence gate, and nothing else; a change that widens that scope needs a good reason.
- **Never contribute scroll data, results computed on scroll data or personal data.** That includes sealed
  records and logs of runs on scroll data: a record names the local paths it read and, with the licence gate
  on, the person who accepted the data terms. Tests, examples and bug reproductions use synthetic data only.
- **Do not copy code from incompatibly licensed projects.** villa's `volume-cartographer/` (VC3D) is
  GPL-3.0: its sources may be read to learn a file format and cited by `path:line`, as the existing modules
  do, but not copied. Code ported from an MIT-licensed project is credited in the module that holds it, and
  the project's notice is added to [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Development set-up

You need git and a Python version that `requires-python` in [pyproject.toml](pyproject.toml) allows; CI
runs the tests on every Python version the package classifiers list.

```sh
git clone https://github.com/LaBlueberryPie/apok-eval.git
cd apok-eval
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test,lzw]"
python -m pip install ruff
```

The `test` extra installs pytest and zarr-python (a reference reader the OME-Zarr tests compare against);
the `lzw` extra installs imagecodecs, for LZW-compressed tifxyz.

## Running the tests

```sh
python -m pytest                          # the whole suite
python -m pytest tests/test_io_tifxyz.py  # one module
python -m pytest -k umbilicus             # tests whose names match a keyword
```

- Every test is synthetic: the test data are built on the fly, and no test reads scroll data.
- A network guard (`tests/netguard.py`, installed by `tests/conftest.py` in the test process and in every
  Python subprocess a test starts) blocks every connection beyond loopback. A test of a remote reader starts
  its own server on loopback.
- Some tests skip, saying why, where the platform or the environment does not allow them: tests that need
  Linux's `/proc`, FIFOs or a git that can create a SHA-256 repository, and the tests of LZW refusals
  without imagecodecs, which skip when imagecodecs is installed. Run those in an environment without the
  `lzw` extra.
- The release guard's scanning tests skip without the maintainers' private files (see below). That is
  expected.

The suite takes a few minutes. CI runs it on every supported Python version for each push and pull request
to `main`, together with the lint and a build of the sdist and the wheel.

## Linting

```sh
ruff check .
```

The rule set, in [pyproject.toml](pyproject.toml), is deliberately conservative: pyflakes and pycodestyle's
error and warning checks. CI pins the ruff version it uses ([.github/workflows/ci.yml](.github/workflows/ci.yml)).
The code is not auto-formatted: do not reformat code your change does not touch.

## The release guards

Two tests protect what this repository must never contain. Both run with the rest of the suite; in CI, the
figure guard's scanning tests skip (see below).

- **No withheld figure** (`tests/test_no_paris4_figures.py`). Results computed on PHerc. Paris 4 data are not
  published here. Before each release, the maintainers run this guard: it reduces every number written in the
  repository (in tracked and staged files, and in every file and message of every commit) to a canonical
  form, using `release/figures.py`, and looks it up in a private list of withheld figures. Numbers that match by
  coincidence, such as a constant or a synthetic test value, are reviewed one line at a time. The private
  list and its allow-list are never committed (`.gitignore` names both): without them, as in any clone and in
  CI, the scanning tests skip, while the tests of the normaliser, and the check that neither private file is
  tracked, staged or committed, always run. [release/README.md](release/README.md) explains the guard and
  what it cannot see.

  What this means for you: never add a figure computed on scroll data, in any form. A line you add or change
  that holds a number may match by coincidence; a maintainer then reviews it and may ask you to reword it.
  Commit messages are scanned too.
- **No private link** (`tests/test_no_private_links.py`). No tracked file and no commit message links to a
  private workspace: such links open only for their owner, and they name private records.

A related guard, `tests/test_licence_hygiene.py`, fails if any tracked file sets `terms_accepted_by`: an
acceptance of data terms belongs to one person and is never committed ([docs/licence.md](docs/licence.md),
"No acceptance is committed").

## Coding style

Match the code around your change. In particular:

- **Modules** open with a docstring and `from __future__ import annotations`. The module docstrings are the
  normative statement of the rules a module applies; the pages in [docs/](docs) follow them. A change of
  behaviour updates the docstring and the matching page in the same commit.
- **Guards refuse; they never guess.** A guard raises `Refusal` (`apok_eval.refusal`) as a literal
  `raise Refusal(...)` statement, with a stable code registered through `register_codes` (upper snake case,
  prefixed by its area: `EVAL_`, `IO_`, `LEDGER_`, `LICENCE_`) and a reason that names the cause. It never
  returns a convenient default in place of a refusal. A new code is documented alongside the other codes of
  its module, and a test makes the guard refuse.
- **Inputs** named by the user are read through `apok_eval.fileio`, and JSON through `apok_eval.jsonstrict`,
  never Python's `json` module directly.
- **No network access** outside `apok_eval.io.store`, and none there without a clearance from the licence
  gate.
- **Tests** build their own synthetic data, seed any randomness, reach nothing beyond loopback and run in a
  few seconds where possible.
- **Spelling** is British: *licence* as the noun (the licence gate, `apok_eval.licence`) and *license* only
  where an external name requires it, such as the `license` field of `pyproject.toml`.
- **Dependencies** are added only after discussion. A new one is listed, with its licence, in the
  dependency table of [README.md](README.md).

## Documentation and changelog

Update [README.md](README.md) and the pages in [docs/](docs) when you change what a command does, a
default, or a refusal. Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for any change a user
would notice.

## Commit messages

- Write the subject line in the imperative mood ("Refuse a mask with more than one page", not "Refused ..."
  or "Refuses ..."), in at most 72 characters, without a trailing full stop.
- Leave a blank line, then explain what changed and why in a body wrapped at 72 characters.
- Keep each commit to one logical change, with its tests and documentation.
- Use `Co-authored-by:` only for people who co-wrote the change.
- A `Signed-off-by:` line is not required.
- Commit messages are scanned by the release guards like every file: no figures computed on scroll data and
  no private links.

## Pull requests

- Branch from `main` and keep each pull request focused on one change.
- Fill in the pull request template, and make sure `python -m pytest` and `ruff check .` pass locally.
- CI must pass before a pull request is merged. A maintainer reviews every pull request and may ask for
  changes.

## Licence

apok-eval is released under the [MIT licence](LICENSE). By contributing, you agree that your contributions
are licensed under the same terms.
