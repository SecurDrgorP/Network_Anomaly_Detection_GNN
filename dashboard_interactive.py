import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import dash_bootstrap_components as dbc

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Network Anomaly Detection Dashboard"

# Load data
@app.callback(Output('dummy', 'children'), [Input('dummy', 'id')])
def load_data():
    """Load data files"""
    try:
        global nodes_df, results_df, edges_train, edges_test
        nodes_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/nodes.csv')
        results_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/results_gnn_predictions.csv')
        edges_train = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_train.csv')
        edges_test = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_test.csv')
        return ""
    except Exception as e:
        return f"Error loading data: {e}"

# Load initial data
try:
    nodes_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/nodes.csv')
    results_df = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/results_gnn_predictions.csv')
    edges_train = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_train.csv')
    edges_test = pd.read_csv('/home/ubuntu/Network_Anomaly_Detection_GNN/data/processed/edges_test.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    nodes_df = pd.DataFrame()
    results_df = pd.DataFrame()
    edges_train = pd.DataFrame()
    edges_test = pd.DataFrame()

# Helper functions
def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate comprehensive metrics"""
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

def create_network_plot():
    """Create network visualization"""
    if edges_train.empty or nodes_df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for _, node in nodes_df.iterrows():
        G.add_node(node['id'], **node.to_dict())
    
    # Add edges (handle different possible column names)
    edge_cols = edges_train.columns.tolist()
    if len(edge_cols) >= 2:
        for _, edge in edges_train.head(500).iterrows():  # Limit for performance
            G.add_edge(edge.iloc[0], edge.iloc[1])
    
    # Generate layout
    pos = nx.spring_layout(G, k=1, iterations=20, seed=42)
    
    # Prepare plot data
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Color nodes based on anomaly status
    node_colors = ['red' if G.nodes[node].get('is_anomaly', 0) else 'lightblue' 
                   for node in G.nodes()]
    
    node_text = [f'Node {node}<br>CPU: {G.nodes[node].get("cpu", 0):.2f}<br>'
                 f'Memory: {G.nodes[node].get("mem", 0):.2f}<br>'
                 f'Anomaly: {bool(G.nodes[node].get("is_anomaly", 0))}'
                 for node in G.nodes()]
    
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
                            marker=dict(color=node_colors, size=8),
                            name='Nodes'))
    
    fig.update_layout(
        title='Network Topology',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🔍 Network Anomaly Detection Dashboard", 
                   className="text-center text-primary mb-4"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Nodes", className="card-title"),
                    html.H2(len(nodes_df), className="text-primary", id="total-nodes")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Anomalies", className="card-title"),
                    html.H2(nodes_df['is_anomaly'].sum() if not nodes_df.empty else 0, 
                            className="text-danger", id="anomaly-count")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Normal Nodes", className="card-title"),
                    html.H2(len(nodes_df) - nodes_df['is_anomaly'].sum() if not nodes_df.empty else 0,
                            className="text-success", id="normal-count")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Anomaly Rate", className="card-title"),
                    html.H2(f"{(nodes_df['is_anomaly'].sum() / len(nodes_df) * 100):.1f}%" 
                            if not nodes_df.empty else "0%",
                            className="text-warning", id="anomaly-rate")
                ])
            ])
        ], width=3),
    ], className="mb-4"),
    
    dbc.Tabs([
        dbc.Tab(label="Data Analysis", tab_id="data-tab"),
        dbc.Tab(label="Model Comparison", tab_id="model-tab"),
        dbc.Tab(label="Network Visualization", tab_id="network-tab"),
        dbc.Tab(label="Performance Metrics", tab_id="metrics-tab"),
    ], id="tabs", active_tab="data-tab"),
    
    html.Div(id="tab-content", className="mt-4"),
    
    # Hidden div for callback triggers
    html.Div(id="dummy", style={"display": "none"})
    
], fluid=True)

# Callback for tab content
@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
def update_tab_content(active_tab):
    if active_tab == "data-tab":
        return create_data_analysis_content()
    elif active_tab == "model-tab":
        return create_model_comparison_content()
    elif active_tab == "network-tab":
        return create_network_content()
    elif active_tab == "metrics-tab":
        return create_metrics_content()
    return html.Div("Select a tab")

def create_data_analysis_content():
    """Create data analysis tab content"""
    if nodes_df.empty:
        return html.Div("No data available")
    
    # Feature distributions
    fig_cpu = px.histogram(nodes_df, x='cpu', color='is_anomaly', 
                          title='CPU Usage Distribution', nbins=30)
    fig_mem = px.histogram(nodes_df, x='mem', color='is_anomaly',
                          title='Memory Usage Distribution', nbins=30)
    
    # Scatter plots
    fig_scatter1 = px.scatter(nodes_df, x='cpu', y='mem', color='is_anomaly',
                             title='CPU vs Memory Usage')
    fig_scatter2 = px.scatter(nodes_df, x='degree', y='cpu', color='is_anomaly',
                             title='Degree vs CPU Usage')
    
    # Correlation matrix
    corr_matrix = nodes_df[['cpu', 'mem', 'degree', 'is_anomaly']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Feature Correlation Matrix")
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_cpu)], width=6),
            dbc.Col([dcc.Graph(figure=fig_mem)], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_scatter1)], width=6),
            dbc.Col([dcc.Graph(figure=fig_scatter2)], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_corr)], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Statistical Summary"),
                dash_table.DataTable(
                    data=nodes_df.describe().round(3).reset_index().to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ['index'] + list(nodes_df.describe().columns)],
                    style_cell={'textAlign': 'left'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                )
            ], width=12)
        ])
    ])

def create_model_comparison_content():
    """Create model comparison tab content"""
    if results_df.empty or 'gnn_prediction' not in results_df.columns:
        return html.Div("No model results available")
    
    y_true = results_df['is_anomaly'].values
    y_pred_gnn = results_df['gnn_prediction'].values
    y_scores_gnn = results_df['gnn_anomaly_score'].values
    
    # Create baseline predictions
    threshold = results_df['cpu'].quantile(0.95)
    baseline_pred = (results_df['cpu'] > threshold).astype(int)
    baseline_scores = results_df['cpu'] / results_df['cpu'].max()
    
    # Calculate metrics
    gnn_metrics = calculate_metrics(y_true, y_pred_gnn, y_scores_gnn)
    baseline_metrics = calculate_metrics(y_true, baseline_pred, baseline_scores)
    
    # Metrics comparison chart
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
        'GNN': [gnn_metrics['accuracy'], gnn_metrics['precision'], 
                gnn_metrics['recall'], gnn_metrics['f1'], gnn_metrics['auc']],
        'Baseline': [baseline_metrics['accuracy'], baseline_metrics['precision'],
                    baseline_metrics['recall'], baseline_metrics['f1'], baseline_metrics['auc']]
    })
    
    fig_metrics = px.bar(metrics_df, x='Metric', y=['GNN', 'Baseline'], 
                        title='Model Performance Comparison', barmode='group')
    
    # ROC curves
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, y_scores_gnn)
    fpr_base, tpr_base, _ = roc_curve(y_true, baseline_scores)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_gnn, y=tpr_gnn, name=f'GNN (AUC = {gnn_metrics["auc"]:.3f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_base, y=tpr_base, name=f'Baseline (AUC = {baseline_metrics["auc"]:.3f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash')))
    fig_roc.update_layout(title='ROC Curve Comparison', xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
    
    # Confusion matrices
    cm_gnn = confusion_matrix(y_true, y_pred_gnn)
    cm_baseline = confusion_matrix(y_true, baseline_pred)
    
    fig_cm_gnn = px.imshow(cm_gnn, text_auto=True, aspect="auto", title='GNN Confusion Matrix')
    fig_cm_baseline = px.imshow(cm_baseline, text_auto=True, aspect="auto", title='Baseline Confusion Matrix')
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("GNN Model Performance"),
                        html.P(f"Accuracy: {gnn_metrics['accuracy']:.3f}"),
                        html.P(f"Precision: {gnn_metrics['precision']:.3f}"),
                        html.P(f"Recall: {gnn_metrics['recall']:.3f}"),
                        html.P(f"F1-Score: {gnn_metrics['f1']:.3f}"),
                        html.P(f"AUC: {gnn_metrics['auc']:.3f}"),
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Baseline Model Performance"),
                        html.P(f"Accuracy: {baseline_metrics['accuracy']:.3f}"),
                        html.P(f"Precision: {baseline_metrics['precision']:.3f}"),
                        html.P(f"Recall: {baseline_metrics['recall']:.3f}"),
                        html.P(f"F1-Score: {baseline_metrics['f1']:.3f}"),
                        html.P(f"AUC: {baseline_metrics['auc']:.3f}"),
                    ])
                ])
            ], width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_metrics)], width=6),
            dbc.Col([dcc.Graph(figure=fig_roc)], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_cm_gnn)], width=6),
            dbc.Col([dcc.Graph(figure=fig_cm_baseline)], width=6),
        ]),
    ])

def create_network_content():
    """Create network visualization tab content"""
    fig_network = create_network_plot()
    
    # Degree distribution
    if not nodes_df.empty:
        fig_degree = px.histogram(nodes_df, x='degree', title='Node Degree Distribution', nbins=20)
    else:
        fig_degree = go.Figure()
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_network)], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Network Statistics"),
                        html.P(f"Total Nodes: {len(nodes_df)}"),
                        html.P(f"Total Edges: {len(edges_train)}"),
                        html.P(f"Average Degree: {nodes_df['degree'].mean():.2f}" if not nodes_df.empty else "N/A"),
                        html.P(f"Network Density: {2 * len(edges_train) / (len(nodes_df) * (len(nodes_df) - 1)):.4f}" 
                               if len(nodes_df) > 1 else "N/A"),
                    ])
                ])
            ], width=6),
            dbc.Col([dcc.Graph(figure=fig_degree)], width=6),
        ]),
    ])

def create_metrics_content():
    """Create detailed performance metrics content"""
    if results_df.empty or 'gnn_prediction' not in results_df.columns:
        return html.Div("No performance data available")
    
    y_true = results_df['is_anomaly'].values
    y_scores = results_df['gnn_anomaly_score'].values
    
    # Threshold analysis
    thresholds = np.linspace(0, 1, 50)
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
    
    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(x=thresholds, y=precision_scores, name='Precision'))
    fig_thresh.add_trace(go.Scatter(x=thresholds, y=recall_scores, name='Recall'))
    fig_thresh.add_trace(go.Scatter(x=thresholds, y=f1_scores, name='F1-Score'))
    fig_thresh.update_layout(title='Performance vs Threshold', xaxis_title='Threshold', yaxis_title='Score')
    
    # Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, name='PR Curve'))
    fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    
    # Score distributions
    fig_scores = px.histogram(results_df, x='gnn_anomaly_score', color='is_anomaly',
                             title='Anomaly Score Distribution', nbins=30)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_thresh)], width=6),
            dbc.Col([dcc.Graph(figure=fig_pr)], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_scores)], width=12),
        ]),
    ])

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)