# Structural Anomaly Detection in Network Topologies using Graph Neural Networks

**Project Subject:** Detection of Structural Anomalies in a Network
**Dataset:** Internet Topology Zoo (Abilene Network)

**NOTE**: you can use another dataset from: [Topology Zoo / Datasets](https://topology-zoo.org/dataset.html)

## 1. Project Abstract

This project addresses the challenge of monitoring modern network infrastructures by moving beyond simple threshold-based alerts. The objective is to detect **structural anomalies**—configurations or connections that deviate from the statistical norm of the network topology—using unsupervised Deep Learning.

We compare two distinct approaches:

1. **DBSCAN (Density-Based Spatial Clustering):** A classical machine learning method analyzing tabular node attributes (CPU, Memory).
2. **Graph Auto-Encoder (GAE):** A Graph Neural Network (GNN) architecture that learns a latent representation of the network topology to identify unlikely connections.

## 2. Technical Architecture

### 2.1. Environment and Prerequisites

The solution is implemented in Python using the following core libraries:

* **PyTorch & PyTorch Geometric:** For GNN implementation and tensor operations.
* **NetworkX:** For graph manipulation and generation.
* **Scikit-Learn:** For the DBSCAN implementation and performance metrics.
* **Pandas/NumPy:** For data preprocessing and feature engineering.

### 2.2. Data Pipeline

Since public datasets (Topology Zoo) provide only static topology, a synthetic feature generation pipeline was developed:

* **Topology Source:** Abilene Network (backbone).
* **Feature Simulation:** CPU and Memory usage are simulated based on node degree, with Gaussian noise added to mimic real-world variance.
* **Anomaly Injection:**
* *Attribute Anomalies:* High resource usage injected into low-degree edge nodes.
* *Structural Anomalies:* Non-existent links added between unrelated nodes to simulate misconfigurations.



## 3. Methodology

### Phase 1: Tabular Clustering (Baseline)

We utilized **DBSCAN** to cluster nodes based on normalized CPU and Memory vectors.

* *Hypothesis:* Anomalies appear as low-density points (outliers) in the feature space.
* *Limitation:* This method treats nodes as independent data points, ignoring the adjacency matrix (network links).

### Phase 2: Graph Representation Learning (Proposed Solution)

We implemented a **Graph Auto-Encoder (GAE)** consisting of:

* **Encoder (GCN):** Two Graph Convolutional layers that compress the input graph () into a low-dimensional latent space .
* **Decoder:** A dot-product decoder that attempts to reconstruct the adjacency matrix from .
* **Anomaly Scoring:** The reconstruction loss acts as the anomaly score. Links with low predicted probabilities during decoding are flagged as structural anomalies.

## 4. Evaluation and Metrics

The models were evaluated against a ground-truth label set generated during the simulation phase.

**Key Performance Indicators (KPIs):**

* **ROC-AUC Score:** To measure the global ranking quality of the anomaly scores.
* **Precision/Recall:** To assess the trade-off between false positives and missed anomalies.

**Results Summary:**
The analysis demonstrates that while DBSCAN effectively identifies simple resource overloads (Attribute Anomalies), it fails to detect topological errors. The GNN approach successfully identifies structural anomalies by leveraging the graph structure, confirming the hypothesis that topological context is essential for robust network monitoring.

## 5. Usage Instructions

1. **Installation:**
Uncomment and run the installation commands in the second code cell to install the required dependencies:
```bash
pip install torch torch-geometric networkx pandas scikit-learn matplotlib requests

```


2. **Execution:**
Run the provided Jupyter Notebook `project_gnn_anomaly.ipynb`. The script performs the following automated steps:
* Downloads/Loads the dataset.
* Generates the synthetic feature set and injects anomalies.
* Trains the GAE model (200 epochs).
* Outputs the comparison metrics and the final visualization.


3. **Deliverables:**
* `nodes.csv` & `edges.csv`: The processed dataset.
* `gnn_autoencoder.pth`: The trained model weights.
* Visualizations: Comparisons of DBSCAN clusters vs. GNN structural flags.
