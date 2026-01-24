import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch
import os
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_network
from utils.feature_generator import generate_features_and_anomalies
from utils import models, baseline
import config

# Set page configuration
st.set_page_config(
    page_title="Network Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ModelComparison:
    def __init__(self):
        self.load_data()
        self.load_model()
    
    def load_data(self):
        """Load processed data and results"""
        try:
            # Load processed data
            self.nodes_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'nodes.csv'))
            self.edges_train = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'edges_train.csv'))
            self.edges_test = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'edges_test.csv'))
            self.predictions = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'results_gnn_predictions.csv'))
            
            # Load network structure
            self.G_clean = load_network()
            self.G_clean, self.G_dirty, _, _, _ = generate_features_and_anomalies(self.G_clean)
            
        except FileNotFoundError as e:
            st.error(f"Data files not found: {e}")
            st.stop()
    
    def load_model(self):
        """Load trained model if available"""
        model_path = os.path.join(config.MODELS_DIR, 'gnn_model.pth')
        if os.path.exists(model_path):
            try:
                # Get model architecture
                self.model = models.build_model(self.nodes_df.shape[1] - 2)  # excluding id and is_anomaly
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            except Exception as e:
                st.warning(f"Could not load model: {e}")
                self.model = None
        else:
            self.model = None

def create_sidebar():
    """Create sidebar with model controls"""
    with st.sidebar:
        st.title("🔍 Anomaly Detection Dashboard")
        st.markdown("---")
        
        # Model comparison options
        st.header("Model Comparison")
        compare_models = st.checkbox("Compare Models", value=True)
        
        # Threshold controls
        st.header("Detection Thresholds")
        gnn_threshold = st.slider("GNN Threshold", 0.0, 1.0, 0.5, 0.01)
        dbscan_eps = st.slider("DBSCAN Epsilon", 1.0, 10.0, config.DBSCAN_EPS, 0.1)
        dbscan_min_samples = st.slider("DBSCAN Min Samples", 1, 10, config.DBSCAN_MIN_SAMPLES)
        
        # Visualization options
        st.header("Visualization Options")
        show_network = st.checkbox("Show Network Graph", value=True)
        show_feature_dist = st.checkbox("Show Feature Distributions", value=True)
        
        return {
            'compare_models': compare_models,
            'gnn_threshold': gnn_threshold,
            'dbscan_eps': dbscan_eps,
            'dbscan_min_samples': dbscan_min_samples,
            'show_network': show_network,
            'show_feature_dist': show_feature_dist
        }

def create_metrics_cards(predictions_df):
    """Create metric cards showing model performance"""
    col1, col2, col3, col4 = st.columns(4)
    
    y_true = predictions_df['is_anomaly']
    y_pred_gnn = predictions_df['gnn_prediction']
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred_gnn)
    precision = precision_score(y_true, y_pred_gnn, zero_division=0)
    recall = recall_score(y_true, y_pred_gnn, zero_division=0)
    f1 = f1_score(y_true, y_pred_gnn, zero_division=0)
    
    # Try to calculate AUC if possible
    try:
        auc = roc_auc_score(y_true, predictions_df['gnn_anomaly_score'])
    except:
        auc = 0.0
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

def plot_roc_curve(predictions_df):
    """Plot ROC curve"""
    y_true = predictions_df['is_anomaly']
    y_scores = predictions_df['gnn_anomaly_score']
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc_score(y_true, y_scores):.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=400,
        height=400
    )
    
    return fig

def plot_precision_recall_curve(predictions_df):
    """Plot Precision-Recall curve"""
    y_true = predictions_df['is_anomaly']
    y_scores = predictions_df['gnn_anomaly_score']
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=400,
        height=400
    )
    
    return fig

def plot_confusion_matrix(predictions_df):
    """Plot confusion matrix"""
    y_true = predictions_df['is_anomaly']
    y_pred = predictions_df['gnn_prediction']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        width=400,
        height=400
    )
    
    return fig

def plot_score_distribution(predictions_df):
    """Plot anomaly score distribution"""
    fig = go.Figure()
    
    # Normal nodes
    normal_scores = predictions_df[predictions_df['is_anomaly'] == 0]['gnn_anomaly_score']
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal',
        opacity=0.7,
        nbinsx=30,
        marker_color='blue'
    ))
    
    # Anomalous nodes
    anomaly_scores = predictions_df[predictions_df['is_anomaly'] == 1]['gnn_anomaly_score']
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomaly',
        opacity=0.7,
        nbinsx=30,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Anomaly Score Distribution',
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        barmode='overlay',
        width=800,
        height=400
    )
    
    return fig

def plot_network_graph(G, predictions_df, show_anomalies=True):
    """Plot network graph with anomalies highlighted"""
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node info from predictions
        node_info = predictions_df[predictions_df['id'] == node]
        if len(node_info) > 0:
            is_anomaly = node_info.iloc[0]['is_anomaly']
            gnn_score = node_info.iloc[0]['gnn_anomaly_score']
            cpu = node_info.iloc[0]['cpu']
            
            node_text.append(f'Node {node}<br>CPU: {cpu:.2f}<br>Anomaly Score: {gnn_score:.3f}')
            
            if show_anomalies and is_anomaly:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        else:
            node_text.append(f'Node {node}')
            node_colors.append('lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=10,
            color=node_colors,
            line=dict(width=2, color='DarkSlateGrey')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Network Graph (Red = Anomalies)',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Red nodes indicate detected anomalies",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002 ) ],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def plot_feature_distributions(predictions_df):
    """Plot feature distributions comparing normal vs anomalous nodes"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['CPU Usage', 'Memory Usage', 'Node Degree']
    )
    
    normal_nodes = predictions_df[predictions_df['is_anomaly'] == 0]
    anomaly_nodes = predictions_df[predictions_df['is_anomaly'] == 1]
    
    # CPU distribution
    fig.add_trace(
        go.Histogram(x=normal_nodes['cpu'], name='Normal CPU', opacity=0.7, marker_color='blue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=anomaly_nodes['cpu'], name='Anomaly CPU', opacity=0.7, marker_color='red'),
        row=1, col=1
    )
    
    # Memory distribution
    fig.add_trace(
        go.Histogram(x=normal_nodes['mem'], name='Normal Memory', opacity=0.7, marker_color='blue', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=anomaly_nodes['mem'], name='Anomaly Memory', opacity=0.7, marker_color='red', showlegend=False),
        row=1, col=3
    )
    
    # Degree distribution
    fig.add_trace(
        go.Histogram(x=normal_nodes['degree'], name='Normal Degree', opacity=0.7, marker_color='blue', showlegend=False),
        row=1, col=3
    )
    fig.add_trace(
        go.Histogram(x=anomaly_nodes['degree'], name='Anomaly Degree', opacity=0.7, marker_color='red', showlegend=False),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Feature Distributions: Normal vs Anomalous Nodes',
        height=400,
        barmode='overlay'
    )
    
    return fig

def run_model_comparison(data, params):
    """Run DBSCAN comparison with current parameters"""
    from utils.baseline import run_dbscan_with_params
    
    # Create a subset of node features for DBSCAN
    features_df = data.nodes_df[['cpu', 'mem', 'degree']].copy()
    
    try:
        dbscan_metrics = run_dbscan_with_params(
            features_df, 
            data.nodes_df['is_anomaly'],
            eps=params['dbscan_eps'],
            min_samples=params['dbscan_min_samples']
        )
        return dbscan_metrics
    except Exception as e:
        st.error(f"Error running DBSCAN: {e}")
        return None

def main():
    """Main dashboard function"""
    st.title("🔍 Network Anomaly Detection Dashboard")
    st.markdown("Compare GNN and DBSCAN models for network anomaly detection")
    
    # Load data and model
    data = ModelComparison()
    
    # Create sidebar
    params = create_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Model Performance Overview")
        metrics = create_metrics_cards(data.predictions)
    
    with col2:
        st.header("Dataset Info")
        st.metric("Total Nodes", len(data.nodes_df))
        st.metric("Anomalous Nodes", data.nodes_df['is_anomaly'].sum())
        st.metric("Normal Nodes", (data.nodes_df['is_anomaly'] == 0).sum())
        anomaly_rate = data.nodes_df['is_anomaly'].mean() * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    # Performance plots
    st.header("📊 Model Performance Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        roc_fig = plot_roc_curve(data.predictions)
        st.plotly_chart(roc_fig, use_container_width=True)
    
    with col2:
        pr_fig = plot_precision_recall_curve(data.predictions)
        st.plotly_chart(pr_fig, use_container_width=True)
    
    with col3:
        cm_fig = plot_confusion_matrix(data.predictions)
        st.plotly_chart(cm_fig, use_container_width=True)
    
    # Score distribution
    st.header("📈 Anomaly Score Analysis")
    score_fig = plot_score_distribution(data.predictions)
    st.plotly_chart(score_fig, use_container_width=True)
    
    # Model comparison
    if params['compare_models']:
        st.header("🆚 Model Comparison")
        
        # Run DBSCAN with current parameters
        dbscan_results = run_model_comparison(data, params)
        
        if dbscan_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("GNN Model")
                st.write(f"**Precision:** {metrics['precision']:.3f}")
                st.write(f"**Recall:** {metrics['recall']:.3f}")
                st.write(f"**F1-Score:** {metrics['f1']:.3f}")
                st.write(f"**AUC:** {metrics['auc']:.3f}")
            
            with col2:
                st.subheader("DBSCAN Model")
                st.write(f"**Precision:** {dbscan_results['precision']:.3f}")
                st.write(f"**Recall:** {dbscan_results['recall']:.3f}")
                st.write(f"**F1-Score:** {dbscan_results['f1']:.3f}")
                # DBSCAN doesn't provide probability scores for AUC
                st.write(f"**AUC:** N/A")
            
            # Comparison chart
            comparison_df = pd.DataFrame({
                'Model': ['GNN', 'DBSCAN'],
                'Precision': [metrics['precision'], dbscan_results['precision']],
                'Recall': [metrics['recall'], dbscan_results['recall']],
                'F1-Score': [metrics['f1'], dbscan_results['f1']]
            })
            
            fig = px.bar(
                comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Metric', y='Score', color='Model', barmode='group',
                title='Model Performance Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Network visualization
    if params['show_network']:
        st.header("🌐 Network Visualization")
        network_fig = plot_network_graph(data.G_dirty, data.predictions)
        st.plotly_chart(network_fig, use_container_width=True)
    
    # Feature distributions
    if params['show_feature_dist']:
        st.header("📊 Feature Analysis")
        feature_fig = plot_feature_distributions(data.predictions)
        st.plotly_chart(feature_fig, use_container_width=True)
    
    # Raw data exploration
    st.header("🔍 Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["Predictions", "Node Features", "Edge Data"])
    
    with tab1:
        st.subheader("Model Predictions")
        st.dataframe(data.predictions)
    
    with tab2:
        st.subheader("Node Features")
        st.dataframe(data.nodes_df)
    
    with tab3:
        st.subheader("Edge Information")
        st.write("Training Edges:", len(data.edges_train))
        st.write("Test Edges:", len(data.edges_test))
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Edges (Sample)")
            st.dataframe(data.edges_train.head(10))
        
        with col2:
            st.subheader("Test Edges (Sample)")
            st.dataframe(data.edges_test.head(10))

if __name__ == "__main__":
    main()