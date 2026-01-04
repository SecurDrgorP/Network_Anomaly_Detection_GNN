from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
import config

def run_dbscan(df_nodes):
    print("\n--- Running Baseline (DBSCAN) ---")
    # RAW FEATURES! (0.1 vs 100.0)
    features = df_nodes[['cpu', 'mem']].values
    
    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES)
    clusters = db.fit_predict(features)
    
    y_pred = [1 if c == -1 else 0 for c in clusters]
    y_true = df_nodes['is_anomaly'].values
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
