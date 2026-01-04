import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Network Settings (Strict VLANs)
NUM_VLANS = 4
NODES_PER_VLAN = 50
PROB_INTRA = 0.5    # Stronger internal connection (easier for GNN to learn)
PROB_INTER = 0.0    # ZERO external traffic (Ground Truth)

# Anomaly Injection
PCT_ATTRIBUTE_ANOMALY = 0.05  # 5% Nodes with High CPU
NUM_STRUCTURAL_ANOMALIES = 25 # Inject 25 Hacker Bridges

# DBSCAN Settings
DBSCAN_EPS = 5.0      # Wide epsilon because we use raw CPU (0.1 vs 100.0)
DBSCAN_MIN_SAMPLES = 2

# GNN Hyperparameters
HIDDEN_DIM = 128      # Wider model to capture VLAN logic better
LATENT_DIM = 64
LEARNING_RATE = 0.01
EPOCHS = 600          # More epochs for deeper convergence
DROPOUT = 0.2
