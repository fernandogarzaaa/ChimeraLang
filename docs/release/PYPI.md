# PyPI Release Runbook

ChimeraLang publishes through PyPI Trusted Publishing from GitHub Actions. This avoids long-lived API tokens and lets PyPI verify that releases came from the repository workflow.

## One-Time PyPI Setup

In your PyPI account, add a pending GitHub trusted publisher:

| Field | Value |
|---|---|
| PyPI project name | `chimeralang` |
| Owner | `fernandogarzaaa` |
| Repository name | `ChimeraLang` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

The project does not need to exist first. PyPI will create it on the first successful publish from the trusted workflow.

## Publish a Release

1. Verify the version in `pyproject.toml`.
2. Push all release commits to `main`.
3. Create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

4. Create a GitHub release for the tag, or run the `Publish to PyPI` workflow manually from GitHub Actions after selecting the release ref.
5. Confirm the package appears at `https://pypi.org/project/chimeralang/`.

## Local Verification

Run these commands before tagging:

```bash
python -m pytest
python -m build
python -m pip install --force-reinstall --no-deps dist/chimeralang-0.1.0-py3-none-any.whl
python -m chimera.cli check examples/mnist_classifier.chimera
```
