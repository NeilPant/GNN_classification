# GNN Classification

A hands-on collection of Jupyter notebooks exploring Graph Neural Networks (GNNs) for classification tasks. This project focuses on node- and graph-level classification using approachable, reproducible notebooks you can run locally or in the cloud.

> If you're viewing this on GitHub, use the notebooks to learn, experiment, and extend. Contributions and suggestions are welcome!

---
## Key Features

- Clean, didactic workflow: data loading → preprocessing → model → training → evaluation
- Reproducible runs with consistent seeds and configuration cells
- CPU-friendly by default; optional GPU acceleration if available
- Modular design to plug in different GNN layers (GCN, GraphSAGE, GAT, etc.)

---
## Quickstart

1) Clone the repo
```bash
git clone https://github.com/NeilPant/GNN_classification.git
cd GNN_classification
```

2) Create and activate an environment (Conda recommended)
```bash
# Create
conda create -n gnn-classification python=3.10 -y
conda activate gnn-classification

# Install PyTorch (choose version for your platform/GPU)
# CPU example (Linux/macOS/Windows):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or see: https://pytorch.org/get-started/locally/
```

3) Install GNN and notebook tooling
- If using PyTorch Geometric:
```bash
# Install PyG (visit https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for your CUDA/CPU settings)
pip install torch-geometric
# Some datasets/transforms may require:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])").html
```

- Common utilities:
```bash
pip install jupyterlab ipywidgets numpy scipy scikit-learn matplotlib seaborn pandas tqdm
```

4) Launch Jupyter and open a notebook
```bash
jupyter lab
# or
jupyter notebook
```

5) Run the cells top-to-bottom to reproduce results.

---

## Environment Setup

Prefer a locked environment? Create an `environment.yml` or `requirements.txt` later and pin versions you’ve tested. Example snippets:

Conda (environment.yml):
```yaml
name: gnn-classification
channels:
  - pytorch
  - pyg
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - pip
  - pip:
      - torch-geometric
      - jupyterlab
      - ipywidgets
      - numpy
      - scipy
      - scikit-learn
      - matplotlib
      - seaborn
      - pandas
      - tqdm
```

Pip (requirements.txt):
```
torch
torch-geometric
jupyterlab
ipywidgets
numpy
scipy
scikit-learn
matplotlib
seaborn
pandas
tqdm
```

Note: Installation steps for PyTorch and PyTorch Geometric can vary by OS and CUDA. Always refer to their official install docs if you hit a snag.

---

## Dataset
i've used Social circles: Facebook dataset from https://snap.stanford.edu to classify users into 3 categories(active/passive/inactive) using a GNN. ive also commented on my thinking and how the code works in the main.py file.

The notebooks are designed to work with commonly used benchmark datasets. Typical sources:


---

## How to Use the Notebooks

- Start with the node classification notebook to understand the basic GNN pipeline.
- Proceed to the graph classification notebook for whole-graph labels.
- includes:
  - Config cell (random seed, hyperparameters, device)
  - Data loading block
  - Model definition 
  - Training/evaluation loop with metrics (accuracy, etc.)
  - Visualization (loss curves, embeddings, confusion matrices)

Tip: If you have a GPU, ensure `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## Reproducibility

For consistent results across runs, notebooks set a global seed. Example code pattern:

```python
import os, random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
```

Note: Some variability may remain due to parallelism and third-party ops.

---
## Extending the Project

- Try alternative architectures:
  - Swap `GCNConv` with `SAGEConv`, `GATConv`, or `GINConv`
- Add regularization:
  - Dropout, weight decay, early stopping
- Hyperparameter search:
  - Grid/random search over hidden dims, layers, LR, dropout
- New datasets:
  - Replace the dataset loading cell with your own data reader
  - Ensure you construct `Data` objects (if using PyG) with `x`, `edge_index`, and labels
- Interpretability:
  - Visualize node embeddings, attention weights (for GAT), or saliency maps

---

## Troubleshooting

- Installation issues with PyTorch Geometric:
  - Consult the official installation matrix for your OS, Python, PyTorch, and CUDA versions
- GPU not detected:
  - Ensure the correct CUDA-enabled PyTorch build is installed; fallback to CPU if needed
- Notebook memory errors:
  - Reduce batch size, hidden dimensions, or number of layers
- Dataset not found / download blocked:
  - Manually download and set the dataset root path to a local folder with appropriate permissions

---

## References

- Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
- Hamilton et al. (2017): Inductive Representation Learning on Large Graphs (GraphSAGE)
- Veličković et al. (2018): Graph Attention Networks (GAT)
- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

If you use this repository in academic work, please cite the relevant papers above.

---

## License

MIT

---

## Acknowledgments

- Open-source contributors to PyTorch and PyTorch Geometric
- Classic GNN benchmark datasets and their maintainers

---
Feel free to open a discussion or pull request with suggestions and improvements.
