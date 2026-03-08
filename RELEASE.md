# Release: nwf-core

## Publish to PyPI

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
