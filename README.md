# nwf-core

[![PyPI version](https://badge.fury.io/py/nwf-core.svg)](https://pypi.org/project/nwf-core/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/romero19912017-ui/nwf-core/actions/workflows/test.yml/badge.svg)](https://github.com/romero19912017-ui/nwf-core/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is NWF?

**Neural Weight Fields (NWF)** is a principled approach to representing data and knowledge in machine learning. Instead of point embeddings, each object is encoded as a semantic charge **(z, Σ)**:

- **z** — position vector in latent space (semantic core)
- **Σ** — diagonal covariance matrix expressing model uncertainty about the object

This pair induces a **semantic potential** — a function decaying with distance. A collection of charges forms a **semantic field** — a continuous measure of data density in latent space.

### Why NWF?

- **Incremental learning without catastrophic forgetting** — add new classes by simply adding new charges to the index; no retraining, no knowledge erasure
- **Built-in uncertainty** — covariance Σ enables calibrated confidence estimates and out-of-distribution (OOD) detection
- **Scalability** — charges can be indexed with FAISS, HNSW (planned) for millions of objects

---

## Installation

```bash
pip install nwf-core
# With FAISS: pip install nwf-core[faiss]
# With VAE (torch): pip install nwf-core[torch]
# All: pip install nwf-core[all]
```

---

## Components

### Charge
Base data structure: center `z` and diagonal covariance `sigma`.

- `to_dict()` / `from_dict()` — JSON serialization
- `to_vector()` — concatenated `[z, log(sigma)]` for indexing
- `whiten(mean, std)` — transform to whitened space
- `clip_sigma(min_val)` — ensure minimum sigma values

### Field
Container for charges with labels and ids.

- `add(charges, labels, ids)` — add charges
- `remove(ids)` — remove by id
- `search(query, k)` — k-nearest search (symmetric Mahalanobis)
- `save(path)` / `load(path)` — persist to disk
- `__iter__` — iterate over (charge, label, id)
- `__getitem__(idx)` — get charge by index

### Metrics
- `mahalanobis_symmetric(z1, sigma1, z2, sigma2)` — symmetric Mahalanobis distance
- `euclidean(z1, z2)` — L2 distance
- `cosine(z1, z2)` — cosine distance

### Potential (OOD detection)
- `potential(r, charges)` — semantic potential at point r; higher = in-distribution
- `potential_batch(r, z_all, sigma_all)` — batch version for multiple queries

### Indices
- **BruteForceIndex** — exact search for small datasets; supports `l2`, `cosine`; batch search
- **FAISSIndex** — FAISS-backed; metrics `l2`, `cosine`, `ip`; optional two-stage Mahalanobis reranking; save/load

### Calibrators
- **AgreementRatio** — fraction of k-nearest neighbors agreeing with prediction (no training)
- **PlattScaler** — logistic regression for probability calibration

### Encoders (optional, requires torch)
- **VAEEncoder** — MLP VAE for flat vectors; outputs `(z, sigma)`

---

## Quick start

```python
import numpy as np
from nwf import Charge, Field, mahalanobis_symmetric

# Create charges
c1 = Charge(z=np.array([0.0, 0.0]), sigma=np.array([0.1, 0.1]))
c2 = Charge(z=np.array([1.0, 1.0]), sigma=np.array([0.1, 0.1]))

# Build field and search
field = Field()
field.add([c1, c2], labels=[0, 1])
distances, indices, labels = field.search(c1, k=2)
```

See [documentation](https://romero19912017-ui.github.io/nwf-core/) and [nwf-vision](https://github.com/romero19912017-ui/nwf-vision) for examples.

---

## Examples

Install with examples dependencies:
```bash
pip install nwf-core[examples]
```

| Script | Description |
|--------|-------------|
| [quickstart.py](examples/quickstart.py) | Minimal workflow: synthetic 2D data, Field, k-NN, AgreementRatio |
| [calibration_demo.py](examples/calibration_demo.py) | Confidence calibration: AgreementRatio, PlattScaler, reliability diagram, ECE |
| [potential_ood.py](examples/potential_ood.py) | OOD detection via semantic potential: histograms, ROC, AUC |

Run:
```bash
python examples/quickstart.py --k 5
python examples/calibration_demo.py --save results/cal.png
python examples/potential_ood.py --save results/ood.png
```

Notebooks in `notebooks/` mirror these examples.

---

## Application areas (сферы применения)

| Area | Use case | Components |
|------|----------|------------|
| **Continual / incremental learning** | Add new classes without retraining; no catastrophic forgetting | Field, Charge, k-NN |
| **Calibrated predictions** | Reliable confidence scores; reliability diagrams, ECE | AgreementRatio, PlattScaler |
| **OOD detection** | Flag out-of-distribution inputs; safety-critical systems | potential, potential_batch |
| **Active learning** | Uncertainty-based sample selection for labeling | trace(sigma), Field |
| **Retrieval** | Semantic search with uncertainty-aware distance | FAISSIndex, mahalanobis_symmetric |

---

## Development

```bash
pip install -e ".[dev,docs]"
pre-commit install
make lint
make format
make html
```

---

## Links

- **Article (Habr):** [Нейровесовые Поля (NWF)](https://github.com/romero19912017-ui/nwf-research/blob/main/HABR.md) — теория и эксперименты
- **Research repo:** [nwf-research](https://github.com/romero19912017-ui/nwf-research)

## License

MIT

---

# nwf-core (Русский)

## Что такое NWF?

**Neural Weight Fields (Нейровесовые поля)** — подход к представлению данных в машинном обучении. Каждый объект кодируется **семантическим зарядом** **(z, Σ)**:

- **z** — вектор положения в латентном пространстве (семантическое ядро)
- **Σ** — диагональная ковариационная матрица (неопределённость модели)

Совокупность зарядов создаёт **семантическое поле** — непрерывную меру плотности данных.

### Преимущества NWF

- **Инкрементальность** — добавление новых классов без переобучения и забывания
- **Встроенная неопределённость** — калибровка уверенности и OOD-детекция
- **Масштабируемость** — индексация через FAISS для миллионов объектов

## Установка

```bash
pip install nwf-core
# С FAISS: pip install nwf-core[faiss]
# С VAE: pip install nwf-core[torch]
```

## Компоненты

- **Charge** — заряд (z, sigma), сериализация, whiten, clip_sigma
- **Field** — контейнер зарядов, add/remove/search, save/load
- **Метрики** — mahalanobis_symmetric, euclidean, cosine
- **Потенциал** — potential, potential_batch для OOD-детекции
- **Индексы** — BruteForceIndex, FAISSIndex (l2, cosine, ip; rerank по Махаланобису)
- **Калибраторы** — AgreementRatio, PlattScaler
- **VAEEncoder** — MLP VAE (опционально)

## Лицензия

MIT
