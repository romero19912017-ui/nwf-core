# Contributing to nwf-core

Thank you for your interest in contributing to NWF.

## How to contribute

### Reporting bugs

- Use the [GitHub issue tracker](https://github.com/romero19912017-ui/nwf-core/issues)
- Describe the problem, steps to reproduce, and your environment (Python version, OS)
- Include a minimal code snippet if possible

### Proposing changes

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature` or `fix/your-fix`
3. Make your changes
4. Run tests: `pytest tests -v`
5. Run linters: `make lint` (or `black src tests`, `isort src tests`, `flake8 src tests`)
6. Commit with a clear message
7. Open a Pull Request

### Code requirements

- Python 3.9+
- Code style: black (line length 88), isort (profile black), flake8
- All new code should have tests
- Docstrings for public API

### Development setup

```bash
pip install -e ".[dev,docs]"
pre-commit install
make lint   # check
make format # auto-format
make html   # build docs
```

### Review process

- PRs require at least one maintainer review
- CI must pass (tests, lint)
- We may ask for changes before merging
