# GNN_classification
i've used Social circles: Facebook dataset from https://snap.stanford.edu to classify users into 3 categories(active/passive/inactive) using a GNN. ive also commented on my thinking and how the code works in the main.py file.
UPDATE:
Overview
Built a compact GCN in PyTorch Geometric to classify nodes as Active/Passive/Inactive on the Ego‑Facebook graph. 🧠📊

Achieved up to ~88% accuracy with a reproducible, hardware‑aware training pipeline. ✅⚡

Data and features
Loaded an undirected NetworkX graph from facebook_combined.txt (Stanford dataset), remapped nodes to contiguous IDs, and built a bidirectional COO edge_index for PyG Data(x, edge_index, y). 📥🔗

Engineered and standardized five node features to boost signal: normalized degree, clustering coefficient, PageRank, sampled betweenness centrality, and average neighbor degree. 🧮✨

Derived labels from degree buckets to represent activity levels: Active, Passive, Inactive. 🏷️📈

Model architecture
Stacked three GCNConv layers with BatchNorm, ReLU, and 0.5 dropout for stable, regularized learning. 🧱🧪

Disabled add_self_loops since preprocessed edges already handled diagonal connections. 🚫🔁

Kept the model lean with hidden size H=32, totaling ~1.5K parameters for fast iteration and deployability. 💡⚙️

Training pipeline
Ensured reproducibility via deterministic seeds across libraries and frameworks. 🎯🔒

Used AdamW with weight decay and cosine annealing LR scheduling to balance speed and generalization. 🧭🛠️

Added optional AMP scaffolding for mixed‑precision training on supported GPUs; included an alternative aot_eager compile path for tracing without Inductor codegen. 🚀🧩

Splits and evaluation
Employed stratified train/test splits to preserve class balance across labels. 📚⚖️

Tracked epoch‑wise loss, train accuracy, and test accuracy to monitor generalization and early stabilization. 📈📝

Observed test accuracy ranging 70–88% depending on epoch budget and hyperparameters; a 200‑epoch run reached ~88%, while earlier versions were ~70%. 🎯⏱️

Overfitting controls
Reduced overfitting using BatchNorm after each hidden GCNConv, ReLU activations, and 0.5 dropout. 🧪🛡️

Architectural and data preprocessing choices (e.g., loop handling) further stabilized training. 🧩🏗️

Class distribution
Reported label skew to contextualize metrics: Active 76.2%, Passive 21.9%, Inactive 1.9%. 📊🔎

Reproducibility and MLOps
Documented configuration (paths, hyperparameters, seeds) and modularized feature engineering plus train/eval steps. 📚🧱

Organized code into clear cells: data loading, feature engineering, model, training loop, and evaluation plots for smooth handoff. 🗂️🧭

Included loss/accuracy plots and total parameter count to evidence compactness and readiness. 📉✅

Deployment
Serialized and saved weights to final_optimized_gnn_with_visualization.pth for downstream inference. 💾🚀

Provided documentation and structure suitable for integration and deployment pipelines. 🛠️📦

One‑line summary
A compact, well‑regularized GCN with engineered graph features delivers up to ~88% accuracy on Ego‑Facebook, with robust reproducibility, monitoring, and deployment readiness. 🧠✅


W
