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

def run_dbscan_with_params(features_df, y_true, eps=None, min_samples=None):
    """Run DBSCAN with custom parameters for dashboard comparison"""
    if eps is None:
        eps = config.DBSCAN_EPS
    if min_samples is None:
        min_samples = config.DBSCAN_MIN_SAMPLES
    
    # Use all available features
    features = features_df.values
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(features)
    
    # Convert clusters to binary anomaly predictions (-1 = anomaly, others = normal)
    y_pred = [1 if c == -1 else 0 for c in clusters]
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'predictions': y_pred
    }
