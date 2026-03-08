Installation
============

Requirements
------------

- Python 3.9+
- numpy >= 1.21
- scipy >= 1.7
- scikit-learn >= 1.0

Basic install
-------------

.. code-block:: bash

   pip install nwf-core

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/romero19912017-ui/nwf-core
   cd nwf-core
   pip install -e .

Optional dependencies
---------------------

- **FAISS** (fast approximate search): ``pip install nwf-core[faiss]``
- **PyTorch** (VAE encoder): ``pip install nwf-core[torch]``
- **All**: ``pip install nwf-core[all]``
