# Structural Anomaly Detection in Network Topologies using Graph Neural Networks

## Overview

Modern network infrastructures (cloud, ISP backbones, enterprise VLANs, SOC environments) face increasingly complex failure and attack patterns. Traditional monitoring systems rely heavily on **threshold-based alerts** (CPU, memory, bandwidth), which are insufficient to detect **topological misconfigurations or stealthy lateral connections**.

This project proposes a **graph-based anomaly detection framework** that detects both:

* **Attribute anomalies** (e.g., abnormal resource usage)
* **Structural anomalies** (e.g., unauthorized links between isolated network segments)

by explicitly modeling the **network topology** using **Graph Neural Networks (GNNs)**.

---

## Project Objectives

* Detect **structural anomalies** that cannot be identified using classical tabular methods
* Compare **traditional ML (DBSCAN)** with **Graph Representation Learning**
* Simulate a **realistic secure network scenario** (strict VLAN isolation)
* Demonstrate why **topological context is essential** for anomaly detection in networks

---

## Dataset

### Base Topology

* **Source:** Internet Topology Zoo (conceptually inspired)
* **Implementation:** Synthetic **VLAN-based network topology**
* **Model:** Stochastic Block Model (SBM)

Each VLAN represents a **secure subnet**, where:

* Intra-VLAN communication is allowed
* Inter-VLAN communication is **strictly forbidden**

This design provides a **clean ground truth** for detecting structural violations.

> 💡 The framework is dataset-agnostic and can be applied to any real network topology provided as an edge list.

---

## Anomaly Types Modeled

### 1. Attribute Anomalies

Simulated as **extreme CPU usage spikes**:

* Normal nodes: CPU ∈ [0.1, 1.0]
* Anomalous nodes: CPU ∈ [90, 100]

These anomalies are designed to be **easily detectable by DBSCAN**, serving as a baseline.

---

### 2. Structural Anomalies (Core Contribution)

Injected as **unauthorized links** between distant VLANs:

* Example: Direct connections between VLAN 0 and VLAN 3
* Represent:

  * Firewall misconfigurations
  * Unauthorized tunnels
  * Lateral movement / backdoors

These anomalies **do not affect node attributes**, making them invisible to classical ML.

---

## Methodology

## Phase 1: Baseline — DBSCAN (Tabular ML)

### Description

* Nodes are treated as independent samples
* Features used:

  * CPU usage
  * Memory usage
* No graph structure is considered

### Hypothesis

Anomalous nodes lie in **low-density regions** of the feature space.

### Limitations

* Ignores adjacency and topology
* Cannot detect structural anomalies
* Fails when anomalies are **purely relational**

---

## Phase 2: Graph-Based Learning — Graph Auto-Encoder (GNN)

### Model Architecture

**Encoder**

* GraphSAGE-based encoder
* Learns node embeddings by aggregating neighborhood information
* Captures:

  * VLAN structure
  * Connectivity patterns
  * Structural regularities

**Decoder**

* Dot-product decoder
* Reconstructs the adjacency matrix
* Outputs link existence probabilities

---

### Learning Principle

The model is trained on a **clean topology only**.

At inference time:

* Links that **cannot be reconstructed accurately**
* Are assigned **high reconstruction error**
* And flagged as **structural anomalies**

---

## Anomaly Scoring Strategy

* **Edge-level:** Low reconstructed probability ⇒ suspicious link
* **Node-level:** A node is anomalous if it participates in at least one suspicious link

Final node anomaly score:

```
score(node) = 1 − min(reconstructed_link_probability)
```

---

## Evaluation Protocol

### Ground Truth

* Known injected CPU anomalies
* Known injected inter-VLAN bridges

### Metrics

* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC** (GNN only)

---

## Results Summary

| Method                | Attribute Anomalies | Structural Anomalies | Topology-Aware |
| --------------------- | ------------------- | -------------------- | -------------- |
| DBSCAN                | ✅ Detected          | ❌ Missed             | ❌ No           |
| GNN (GraphSAGE + GAE) | ✅ Detected          | ✅ Detected           | ✅ Yes          |

### Key Findings

* DBSCAN performs well **only** when anomalies affect raw features
* GNN successfully detects **stealth structural violations**
* Structural context is **critical** for robust network anomaly detection

---

## Project Structure

```
Network_Anomaly_Detection/
│
├── data/
│   ├── raw/                # Clean topology
│   └── processed/          # Nodes, edges, predictions
│
├── utils/
│   ├── data_loader.py
│   ├── feature_generator.py
│   ├── dataset.py
│   ├── models.py
│   ├── baseline.py
│   ├── train.py
│   └── visualization.py
│
├── notebooks/
│   └── NADGNN.ipynb
│
├── models/
│   └── gnn_model.pth
│
├── output/
│   ├── dashboard.png
│   └── risk_map.png
│
├── config.py
├── main.py
├── run.sh
└── requirements.txt
```

---

## Usage Instructions

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
./run.sh
```

This will:

* Generate the network topology
* Inject anomalies
* Train the GNN
* Evaluate DBSCAN vs GNN
* Save results, metrics, and visualizations

---

## Outputs & Deliverables

### Data

* `nodes.csv` — node features + ground truth
* `edges_train.csv` — clean topology
* `edges_test.csv` — topology with anomalies
* `results_gnn_predictions.csv` — final scores & predictions

### Models

* `gnn_model.pth` — trained Graph Auto-Encoder

### Visualizations

* **Dashboard:** Training loss, ROC, confusion matrix, metrics comparison
* **Risk Map:** Network visualization with detected anomalous links

---

## Applications

* SOC automation & zero-trust validation
* Cloud network misconfiguration detection
* ISP backbone monitoring
* Insider threat & lateral movement detection
* Digital twin simulation of secure networks

---

## Key Takeaway

> **Anomalies in networks are not always about “high values” —
> they are often about “wrong connections.”**

Graph Neural Networks provide the necessary inductive bias to **understand and protect network structure**, making them indispensable for next-generation network security and monitoring systems.
