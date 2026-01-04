import os
import torch
import numpy as np
from tabulate import tabulate
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score
import config
from utils import data_loader, feature_generator, dataset, baseline, models, train, visualization

def main():
    print(f"--- [Main] Starting Analysis... ---")
    
    # 1. Load Data
    G_clean = data_loader.load_network()
    G_clean, G_dirty, df_nodes, df_edges_train, df_edges_test = feature_generator.generate_features_and_anomalies(G_clean)
    
    # --- SAVE 1: RAW DATASETS ---
    if not os.path.exists(config.PROCESSED_DIR): os.makedirs(config.PROCESSED_DIR)
    print(f"[System] Saving raw processed data to {config.PROCESSED_DIR}...")
    
    # Save Node Features (with ground truth labels)
    df_nodes.to_csv(os.path.join(config.PROCESSED_DIR, 'nodes.csv'), index=False)
    
    # Save Edge Lists (Train & Test)
    df_edges_train.to_csv(os.path.join(config.PROCESSED_DIR, 'edges_train.csv'), index=False)
    df_edges_test.to_csv(os.path.join(config.PROCESSED_DIR, 'edges_test.csv'), index=False)
    
    print("[System] Raw data saved (nodes.csv, edges_train.csv, edges_test.csv).")

    # 2. DBSCAN
    db_metrics = baseline.run_dbscan(df_nodes)
    
    # 3. Train GNN
    train_data = dataset.create_pyg_data(df_nodes, df_edges_train)
    model = models.build_model(train_data.num_features)
    model, losses = train.train_model(model, train_data)
    
    # 4. Evaluate
    test_data = dataset.create_pyg_data(df_nodes, df_edges_test)
    model.eval()
    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)
        adj_pred = model.decoder.forward_all(z)
        
    node_scores = []
    for i in range(len(df_nodes)):
        neighbors = list(G_dirty.neighbors(i))
        if not neighbors:
            node_scores.append(0)
            continue
        link_probs = [adj_pred[i, n].item() for n in neighbors]
        node_scores.append(1.0 - min(link_probs))
        
    y_true = df_nodes['is_anomaly'].values
    prec, rec, thresholds = precision_recall_curve(y_true, node_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_thresh = thresholds[np.argmax(f1)]
    y_pred_gnn = [1 if s > best_thresh else 0 for s in node_scores]
    
    gnn_metrics = {
        'precision': precision_score(y_true, y_pred_gnn),
        'recall': recall_score(y_true, y_pred_gnn),
        'f1': f1_score(y_true, y_pred_gnn),
        'auc': roc_auc_score(y_true, node_scores)
    }

    # --- SAVE 2: FINAL RESULTS WITH PREDICTIONS ---
    print("[System] Saving analysis results...")
    df_results = df_nodes.copy()
    df_results['gnn_anomaly_score'] = node_scores
    df_results['gnn_prediction'] = y_pred_gnn
    df_results.to_csv(os.path.join(config.PROCESSED_DIR, 'results_gnn_predictions.csv'), index=False)
    print("[System] Analysis results saved (results_gnn_predictions.csv).")

    # 5. Report
    print("\n" + "="*40)
    print("      METHOD COMPARISON REPORT      ")
    print("="*40)
    table = [
        ["Metric", "DBSCAN", "GraphSAGE (GNN)"],
        ["Precision", f"{db_metrics['precision']:.4f}", f"{gnn_metrics['precision']:.4f}"],
        ["Recall", f"{db_metrics['recall']:.4f}", f"{gnn_metrics['recall']:.4f}"],
        ["F1-Score", f"{db_metrics['f1']:.4f}", f"{gnn_metrics['f1']:.4f}"],
        ["ROC-AUC", "N/A", f"{gnn_metrics['auc']:.4f}"]
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
    # 6. Save Visualizations
    visualization.save_dashboard(losses, y_true, y_pred_gnn, node_scores, db_metrics, gnn_metrics)
    visualization.plot_network(G_dirty, adj_pred, best_thresh)
    
    if not os.path.exists(config.MODELS_DIR): os.makedirs(config.MODELS_DIR)
    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, 'gnn_model.pth'))
    print(f"[System] Model saved.")

if __name__ == "__main__":
    main()
