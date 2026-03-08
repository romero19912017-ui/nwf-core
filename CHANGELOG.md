# Changelog

## [0.3.0] - 2026-03

### Added
- **Charge.alpha** — weight for superposition (axiom 4). Default 1.0, full backward compatibility.
- **Charge.with_alpha(new_alpha)** — create copy with different weight.
- **potential**, **potential_batch** — now use alpha: Phi = sum_i alpha_i * exp(-0.5*d_i^2).
- **potential_batch** — optional `alpha_all` parameter for weighted batch queries.
- **Field save/load** — stores and restores alpha; backward compatible (missing alphas default to 1.0).
- **weighted_demo.py** — example: weighted voting vs standard, potential with alpha.

### Changed
- Charge: to_dict/from_dict include alpha.
- Charge.from_vector(v, alpha=1.0) — optional alpha for load compatibility.

## [0.2.2] - 2025-03

- Updated README (EN + RU): full NWF description, components, use cases
- Added author email in pyproject.toml
- Expanded pyproject description

## [0.2.1] - Infrastructure

- Sphinx documentation (docs/, installation, user guide, API)
- CI: lint.yml (black, isort, flake8)
- CI: docs.yml (GitHub Pages deploy)
- Charge.clip_sigma(min_val=1e-6)
- Field: __iter__, __getitem__
- BruteForceIndex.search_batch(query_vectors, k)
- pyproject: docs, dev extras; Makefile for lint/format/docs

## [0.2.0] - 2026-03

### Added
- FAISSIndex: FAISS-based ANN with l2, cosine, ip metrics
- Two-stage Mahalanobis reranking for FAISSIndex (rerank=True)
- AgreementRatio: confidence from k-NN agreement
- PlattScaler: logistic regression calibration
- VAEEncoder: MLP VAE for (z, sigma) encoding (optional, requires torch)
- scikit-learn dependency for calibration

### Changed
- Version bump to 0.2.0

## [0.1.0] - 2026-03

### Added
- Charge: (z, sigma) with to_dict, from_dict, to_vector, whiten
- Field: container with add, remove, search, save, load
- Metric: mahalanobis_symmetric, euclidean, cosine
- BruteForceIndex: exact search for l2, cosine
