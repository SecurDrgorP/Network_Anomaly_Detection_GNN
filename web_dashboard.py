import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from utils.models import build_model
import config
import os

# Page configuration
st.set_page_config(
    page_title="Network Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .comparison-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Load processed data
        nodes_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/nodes.csv')
        results_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/results_gnn_predictions.csv')
        edges_train = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_train.csv')
        edges_test = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_test.csv')
        
        return nodes_df, results_df, edges_train, edges_test
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate comprehensive metrics for model evaluation"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_scores is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['auc'] = 0.0
    
    return metrics

def create_network_visualization(edges_df, nodes_df, threshold=0.5):
    """Create an interactive network visualization"""
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, node in nodes_df.iterrows():
        G.add_node(node['id'], 
                  cpu=node['cpu'], 
                  mem=node['mem'], 
                  degree=node['degree'],
                  is_anomaly=node['is_anomaly'])
    
    # Add edges
    for _, edge in edges_df.iterrows():
        if 'source' in edge and 'target' in edge:
            G.add_edge(edge['source'], edge['target'])
        elif len(edge) >= 2:
            G.add_edge(edge.iloc[0], edge.iloc[1])
    
    # Generate layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Prepare data for plotly
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Color nodes based on anomaly status
    node_colors = []
    node_text = []
    for node in G.nodes():
        node_data = G.nodes[node]
        is_anomaly = node_data.get('is_anomaly', 0)
        cpu = node_data.get('cpu', 0)
        mem = node_data.get('mem', 0)
        
        if is_anomaly:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
        
        node_text.append(f'Node {node}<br>CPU: {cpu:.2f}<br>Memory: {mem:.2f}<br>Anomaly: {bool(is_anomaly)}')
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines',
                            name='Edges'))
    
    # Add nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers',
                            hoverinfo='text',
                            text=node_text,
                            marker=dict(
                                showscale=False,
                                color=node_colors,
                                size=10,
                                line=dict(width=2)
                            ),
                            name='Nodes'))
    
    fig.update_layout(title=dict(text='Network Topology', font=dict(size=16)),
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20,l=5,r=5,t=40),
                     annotations=[ dict(
                         text="Red nodes are anomalies",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002,
                         xanchor='left', yanchor='bottom',
                         font=dict(color="red", size=12))
                     ],
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(cm, 
                    text_auto=True,
                    aspect="auto",
                    title=f'{model_name} - Confusion Matrix',
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale='Blues')
    
    return fig

def create_roc_curve_plot(y_true, y_scores, model_names):
    """Create ROC curve comparison"""
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (scores, name) in enumerate(zip(y_scores, model_names)):
        if scores is not None and len(scores) > 0:
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC = {roc_auc:.3f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='black')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def create_metrics_comparison_chart(metrics_dict):
    """Create metrics comparison bar chart"""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    fig = go.Figure()
    
    x = list(range(len(metric_names)))
    width = 0.35
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        fig.add_trace(go.Bar(
            x=[j + (i - len(models)/2 + 0.5) * width for j in x],
            y=values,
            name=model,
            width=width
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis=dict(
            title='Metrics',
            tickvals=x,
            ticktext=[m.capitalize() for m in metric_names]
        ),
        yaxis_title='Score',
        showlegend=True,
        barmode='group'
    )
    
    return fig

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🔍 Network Anomaly Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    nodes_df, results_df, edges_train, edges_test = load_data()
    
    if nodes_df is None:
        st.error("Failed to load data files. Please check if all required files exist.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section", 
                               ["Overview", "Data Analysis", "Model Comparison", 
                                "Network Visualization", "Performance Metrics"])
    
    if page == "Overview":
        show_overview(nodes_df, results_df)
    elif page == "Data Analysis":
        show_data_analysis(nodes_df, results_df)
    elif page == "Model Comparison":
        show_model_comparison(results_df)
    elif page == "Network Visualization":
        show_network_visualization(edges_train, nodes_df)
    elif page == "Performance Metrics":
        show_performance_metrics(results_df)

def show_overview(nodes_df, results_df):
    """Display overview section"""
    st.header("📊 Project Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Nodes", len(nodes_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        anomaly_count = nodes_df['is_anomaly'].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Anomalies", anomaly_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        normal_count = len(nodes_df) - anomaly_count
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Normal Nodes", normal_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        anomaly_rate = (anomaly_count / len(nodes_df)) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cpu = px.histogram(nodes_df, x='cpu', color='is_anomaly', 
                              title='CPU Usage Distribution',
                              labels={'is_anomaly': 'Is Anomaly'})
        st.plotly_chart(fig_cpu, width='stretch')
    
    with col2:
        fig_mem = px.histogram(nodes_df, x='mem', color='is_anomaly',
                              title='Memory Usage Distribution',
                              labels={'is_anomaly': 'Is Anomaly'})
        st.plotly_chart(fig_mem, width='stretch')

def show_data_analysis(nodes_df, results_df):
    """Display data analysis section"""
    st.header("📈 Data Analysis")
    
    # Feature correlations
    st.subheader("Feature Correlations")
    corr_matrix = nodes_df[['cpu', 'mem', 'degree', 'is_anomaly']].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        text_auto=True,
                        aspect="auto",
                        title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, width='stretch')
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter1 = px.scatter(nodes_df, x='cpu', y='mem', 
                                 color='is_anomaly',
                                 title='CPU vs Memory Usage',
                                 labels={'is_anomaly': 'Is Anomaly'})
        st.plotly_chart(fig_scatter1, width='stretch')
    
    with col2:
        fig_scatter2 = px.scatter(nodes_df, x='degree', y='cpu',
                                 color='is_anomaly',
                                 title='Degree vs CPU Usage',
                                 labels={'is_anomaly': 'Is Anomaly'})
        st.plotly_chart(fig_scatter2, width='stretch')
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(nodes_df.describe(), width='stretch')

def show_model_comparison(results_df):
    """Display model comparison section"""
    st.header("🤖 Model Comparison")
    
    if 'gnn_prediction' in results_df.columns and 'gnn_anomaly_score' in results_df.columns:
        y_true = results_df['is_anomaly'].values
        y_pred_gnn = results_df['gnn_prediction'].values
        y_scores_gnn = results_df['gnn_anomaly_score'].values
        
        # Calculate metrics
        gnn_metrics = calculate_metrics(y_true, y_pred_gnn, y_scores_gnn)
        
        # Create baseline predictions if not available
        if 'baseline_prediction' not in results_df.columns:
            # Simple baseline: mark high CPU as anomalies
            threshold = results_df['cpu'].quantile(0.95)
            baseline_pred = (results_df['cpu'] > threshold).astype(int)
            baseline_scores = results_df['cpu'] / 100.0  # Normalize CPU scores
        else:
            baseline_pred = results_df['baseline_prediction'].values
            baseline_scores = results_df.get('baseline_score', results_df['cpu'] / 100.0).values
        
        baseline_metrics = calculate_metrics(y_true, baseline_pred, baseline_scores)
        
        # Display metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 GNN Model")
            for metric, value in gnn_metrics.items():
                st.metric(metric.upper(), f"{value:.3f}")
        
        with col2:
            st.subheader("📊 Baseline Model")
            for metric, value in baseline_metrics.items():
                st.metric(metric.upper(), f"{value:.3f}")
        
        # Comparison charts
        st.subheader("Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics comparison
            metrics_dict = {'GNN': gnn_metrics, 'Baseline': baseline_metrics}
            fig_metrics = create_metrics_comparison_chart(metrics_dict)
            st.plotly_chart(fig_metrics, width='stretch')
        
        with col2:
            # ROC comparison
            fig_roc = create_roc_curve_plot(y_true, [y_scores_gnn, baseline_scores], 
                                          ['GNN', 'Baseline'])
            st.plotly_chart(fig_roc, width='stretch')
        
        # Confusion matrices
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cm_gnn = create_confusion_matrix_plot(y_true, y_pred_gnn, 'GNN')
            st.plotly_chart(fig_cm_gnn, width='stretch')
        
        with col2:
            fig_cm_baseline = create_confusion_matrix_plot(y_true, baseline_pred, 'Baseline')
            st.plotly_chart(fig_cm_baseline, width='stretch')
        
        # Detailed classification reports
        st.subheader("Detailed Classification Reports")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("GNN Classification Report:")
            report_gnn = classification_report(y_true, y_pred_gnn, output_dict=True)
            st.json(report_gnn)
        
        with col2:
            st.text("Baseline Classification Report:")
            report_baseline = classification_report(y_true, baseline_pred, output_dict=True)
            st.json(report_baseline)
    
    else:
        st.error("GNN prediction columns not found in results data.")

def show_network_visualization(edges_df, nodes_df):
    """Display network visualization section"""
    st.header("🕸️ Network Visualization")
    
    # Network topology
    if len(edges_df) > 0:
        fig_network = create_network_visualization(edges_df, nodes_df)
        st.plotly_chart(fig_network, width='stretch')
        
        # Network statistics
        st.subheader("Network Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Edges", len(edges_df))
        
        with col2:
            st.metric("Total Nodes", len(nodes_df))
        
        with col3:
            avg_degree = nodes_df['degree'].mean()
            st.metric("Average Degree", f"{avg_degree:.2f}")
        
        # Degree distribution
        fig_degree = px.histogram(nodes_df, x='degree', 
                                 title='Degree Distribution',
                                 nbins=20)
        st.plotly_chart(fig_degree, width='stretch')
    
    else:
        st.error("No edge data available for visualization.")

def show_performance_metrics(results_df):
    """Display detailed performance metrics"""
    st.header("📊 Performance Metrics")
    
    if 'gnn_prediction' in results_df.columns:
        y_true = results_df['is_anomaly'].values
        y_pred = results_df['gnn_prediction'].values
        y_scores = results_df['gnn_anomaly_score'].values
        
        # Performance over different thresholds
        thresholds = np.linspace(0, 1, 100)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            pred_thresh = (y_scores > threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            prec = precision_score(y_true, pred_thresh, zero_division=0)
            rec = recall_score(y_true, pred_thresh, zero_division=0)
            f1 = f1_score(y_true, pred_thresh, zero_division=0)
            
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
        
        # Plot metrics vs threshold
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(x=thresholds, y=precision_scores, name='Precision'))
        fig_thresh.add_trace(go.Scatter(x=thresholds, y=recall_scores, name='Recall'))
        fig_thresh.add_trace(go.Scatter(x=thresholds, y=f1_scores, name='F1-Score'))
        
        fig_thresh.update_layout(
            title='Performance Metrics vs Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score'
        )
        st.plotly_chart(fig_thresh, width='stretch')
        
        # Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, name='PR Curve'))
        fig_pr.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision'
        )
        st.plotly_chart(fig_pr, width='stretch')
        
        # Error analysis
        st.subheader("Error Analysis")
        
        # False positives and false negatives
        fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**False Positives:**")
            if len(fp_indices) > 0:
                fp_data = results_df.iloc[fp_indices][['id', 'cpu', 'mem', 'degree', 'gnn_anomaly_score']]
                st.dataframe(fp_data.head(10), width='stretch')
            else:
                st.write("No false positives found.")
        
        with col2:
            st.write("**False Negatives:**")
            if len(fn_indices) > 0:
                fn_data = results_df.iloc[fn_indices][['id', 'cpu', 'mem', 'degree', 'gnn_anomaly_score']]
                st.dataframe(fn_data.head(10), width='stretch')
            else:
                st.write("No false negatives found.")

if __name__ == "__main__":
    main()