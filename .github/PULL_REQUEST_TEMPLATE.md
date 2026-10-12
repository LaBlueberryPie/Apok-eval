## Summary

<!-- What does this change do, and why? Link the issue it addresses, for example "Closes #<number>". -->

## How it was tested

<!-- The tests you added or changed, and anything you checked by hand (on synthetic data only). -->

## Checklist

- [ ] `python -m pytest` passes locally.
- [ ] `ruff check .` passes.
- [ ] New behaviour has tests. A new guard raises `Refusal` with a registered code, and a test makes it refuse.
- [ ] Documentation is updated where behaviour, a default or a refusal code changed (module docstrings, `docs/`,
      `README.md`), and `CHANGELOG.md` has an entry under "Unreleased".
- [ ] The change contains no scroll data, no result computed on scroll data, no personal data and no link to a
      private workspace (see "The release guards" in `CONTRIBUTING.md`).
- [ ] Commit messages follow `CONTRIBUTING.md`.
