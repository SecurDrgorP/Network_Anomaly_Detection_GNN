import networkx as nx
import matplotlib
matplotlib.use('Agg') # Force non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config
from sklearn.metrics import roc_curve, confusion_matrix

def save_dashboard(losses, y_true, y_pred_gnn, node_scores, db_metrics, gnn_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.suptitle("Anomaly Detection Dashboard", fontsize=16)

    # 1. Loss
    axes[0,0].plot(losses, color='blue')
    axes[0,0].set_title("Training Loss")
    
    # 2. ROC
    fpr, tpr, _ = roc_curve(y_true, node_scores)
    axes[0,1].plot(fpr, tpr, color='orange', lw=2, label=f'AUC={gnn_metrics["auc"]:.2f}')
    axes[0,1].plot([0,1], [0,1], 'k--')
    axes[0,1].legend()
    axes[0,1].set_title("ROC Curve")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_gnn)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1,0])
    axes[1,0].set_title("Confusion Matrix")

    # 4. Bar Chart
    metrics = ['Prec', 'Rec', 'F1']
    x = range(3)
    db = [db_metrics['precision'], db_metrics['recall'], db_metrics['f1']]
    gnn = [gnn_metrics['precision'], gnn_metrics['recall'], gnn_metrics['f1']]
    axes[1,1].bar([i-0.2 for i in x], db, width=0.4, label='DBSCAN', color='gray')
    axes[1,1].bar([i+0.2 for i in x], gnn, width=0.4, label='GNN', color='green')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metrics)
    axes[1,1].legend()
    axes[1,1].set_title("Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "dashboard.png"))
    print(f"[Visualization] Dashboard saved to {config.OUTPUT_DIR}/dashboard.png")

def plot_network(G, pred_probs, threshold):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    anomalous = [(u, v) for u, v in G.edges() if pred_probs[u, v] < (1.0 - threshold)]
    normal = [(u, v) for u, v in G.edges() if pred_probs[u, v] >= (1.0 - threshold)]
    
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='#3366cc')
    nx.draw_networkx_edges(G, pos, edgelist=normal, alpha=0.1, edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=anomalous, width=2.5, edge_color='red', label='Detected')
    
    plt.legend()
    plt.axis('off')
    plt.savefig(os.path.join(config.OUTPUT_DIR, "risk_map.png"))
    print(f"[Visualization] Map saved to {config.OUTPUT_DIR}/risk_map.png")
