# Release: nwf-core

## Publish to PyPI

If version 0.2.4 is already on PyPI, bump version in pyproject.toml first.

```bash
cd c:\nwf\libraries\nwf-core
pip install build twine
python -m build
twine upload dist/*
```

## Git

```bash
git add examples/ notebooks/ README.md pyproject.toml .gitignore
git status
git commit -m "Add examples: quickstart, calibration_demo, potential_ood; notebooks; README with application areas"
git push origin main
```
