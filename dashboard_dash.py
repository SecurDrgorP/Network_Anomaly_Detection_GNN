import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from utils.data_loader import load_network
from utils.feature_generator import generate_features_and_anomalies
from utils import baseline
import config

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Network Anomaly Detection Dashboard"

# Load data
def load_dashboard_data():
    try:
        # Load processed data
        nodes_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'nodes.csv'))
        edges_train = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'edges_train.csv'))
        edges_test = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'edges_test.csv'))
        predictions = pd.read_csv(os.path.join(config.PROCESSED_DIR, 'results_gnn_predictions.csv'))
        
        # Load network structure
        G_clean = load_network()
        G_clean, G_dirty, _, _, _ = generate_features_and_anomalies(G_clean)
        
        return {
            'nodes_df': nodes_df,
            'edges_train': edges_train,
            'edges_test': edges_test,
            'predictions': predictions,
            'G_clean': G_clean,
            'G_dirty': G_dirty
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

data = load_dashboard_data()

# Create network graph figure
def create_network_graph(G, predictions_df):
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
            
            if is_anomaly:
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
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Network Graph (Red = Anomalies)',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

# Create performance metrics cards
def create_metrics_cards(predictions_df):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_true = predictions_df['is_anomaly']
    y_pred_gnn = predictions_df['gnn_prediction']
    
    accuracy = accuracy_score(y_true, y_pred_gnn)
    precision = precision_score(y_true, y_pred_gnn, zero_division=0)
    recall = recall_score(y_true, y_pred_gnn, zero_division=0)
    f1 = f1_score(y_true, y_pred_gnn, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, predictions_df['gnn_anomaly_score'])
    except:
        auc = 0.0
    
    return [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{accuracy:.3f}", className="card-title"),
                html.P("Accuracy", className="card-text"),
            ])
        ], color="primary", outline=True),
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{precision:.3f}", className="card-title"),
                html.P("Precision", className="card-text"),
            ])
        ], color="success", outline=True),
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{recall:.3f}", className="card-title"),
                html.P("Recall", className="card-text"),
            ])
        ], color="warning", outline=True),
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{f1:.3f}", className="card-title"),
                html.P("F1-Score", className="card-text"),
            ])
        ], color="info", outline=True),
    ]

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🔍 Network Anomaly Detection Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Metrics cards
    dbc.Row([
        dbc.Col(create_metrics_cards(data['predictions'] if data else pd.DataFrame()), width=12)
    ], className="mb-4") if data else html.Div(),
    
    # Controls
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dbc.Label("GNN Threshold:"),
                    dcc.Slider(
                        id='gnn-threshold',
                        min=0, max=1, step=0.01, value=0.5,
                        marks={0: '0', 0.5: '0.5', 1: '1'}
                    ),
                    html.Br(),
                    dbc.Label("DBSCAN Epsilon:"),
                    dcc.Slider(
                        id='dbscan-eps',
                        min=1, max=10, step=0.1, value=config.DBSCAN_EPS,
                        marks={1: '1', 5: '5', 10: '10'}
                    ),
                    html.Br(),
                    dbc.Checklist(
                        id='display-options',
                        options=[
                            {'label': 'Show Network Graph', 'value': 'network'},
                            {'label': 'Show Feature Distributions', 'value': 'features'},
                            {'label': 'Compare Models', 'value': 'compare'}
                        ],
                        value=['network', 'features'],
                        inline=True
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Main content tabs
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id="tabs", value='performance', children=[
                dcc.Tab(label='Model Performance', value='performance'),
                dcc.Tab(label='Network Visualization', value='network'),
                dcc.Tab(label='Feature Analysis', value='features'),
                dcc.Tab(label='Model Comparison', value='comparison'),
                dcc.Tab(label='Data Explorer', value='data')
            ]),
            html.Div(id='tab-content')
        ], width=12)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('gnn-threshold', 'value'),
     Input('dbscan-eps', 'value'),
     Input('display-options', 'value')]
)
def update_tab_content(active_tab, gnn_threshold, dbscan_eps, display_options):
    if not data:
        return html.Div("Error loading data")
    
    if active_tab == 'performance':
        return create_performance_tab()
    elif active_tab == 'network':
        return create_network_tab()
    elif active_tab == 'features':
        return create_features_tab()
    elif active_tab == 'comparison':
        return create_comparison_tab(dbscan_eps)
    elif active_tab == 'data':
        return create_data_tab()

def create_performance_tab():
    # ROC Curve
    y_true = data['predictions']['is_anomaly']
    y_scores = data['predictions']['gnn_anomaly_score']
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc_score(y_true, y_scores):.3f})',
        line=dict(color='blue', width=2)
    ))
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color='green', width=2)
    ))
    pr_fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
    
    # Confusion Matrix
    y_pred = data['predictions']['gnn_prediction']
    cm = confusion_matrix(y_true, y_pred)
    cm_fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        color_continuous_scale='Blues',
        text_auto=True,
        title='Confusion Matrix'
    )
    
    # Score Distribution
    normal_scores = data['predictions'][data['predictions']['is_anomaly'] == 0]['gnn_anomaly_score']
    anomaly_scores = data['predictions'][data['predictions']['is_anomaly'] == 1]['gnn_anomaly_score']
    
    score_fig = go.Figure()
    score_fig.add_trace(go.Histogram(
        x=normal_scores, name='Normal', opacity=0.7, nbinsx=30, marker_color='blue'
    ))
    score_fig.add_trace(go.Histogram(
        x=anomaly_scores, name='Anomaly', opacity=0.7, nbinsx=30, marker_color='red'
    ))
    score_fig.update_layout(title='Anomaly Score Distribution', xaxis_title='Anomaly Score', yaxis_title='Count', barmode='overlay')
    
    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=roc_fig)], width=6),
            dbc.Col([dcc.Graph(figure=pr_fig)], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=cm_fig)], width=6),
            dbc.Col([dcc.Graph(figure=score_fig)], width=6),
        ])
    ])

def create_network_tab():
    network_fig = create_network_graph(data['G_dirty'], data['predictions'])
    return html.Div([
        dcc.Graph(figure=network_fig)
    ])

def create_features_tab():
    predictions_df = data['predictions']
    normal_nodes = predictions_df[predictions_df['is_anomaly'] == 0]
    anomaly_nodes = predictions_df[predictions_df['is_anomaly'] == 1]
    
    # CPU distribution
    cpu_fig = go.Figure()
    cpu_fig.add_trace(go.Histogram(x=normal_nodes['cpu'], name='Normal', opacity=0.7, marker_color='blue'))
    cpu_fig.add_trace(go.Histogram(x=anomaly_nodes['cpu'], name='Anomaly', opacity=0.7, marker_color='red'))
    cpu_fig.update_layout(title='CPU Usage Distribution', xaxis_title='CPU Usage', yaxis_title='Count', barmode='overlay')
    
    # Memory distribution
    mem_fig = go.Figure()
    mem_fig.add_trace(go.Histogram(x=normal_nodes['mem'], name='Normal', opacity=0.7, marker_color='blue'))
    mem_fig.add_trace(go.Histogram(x=anomaly_nodes['mem'], name='Anomaly', opacity=0.7, marker_color='red'))
    mem_fig.update_layout(title='Memory Usage Distribution', xaxis_title='Memory Usage', yaxis_title='Count', barmode='overlay')
    
    # Degree distribution
    degree_fig = go.Figure()
    degree_fig.add_trace(go.Histogram(x=normal_nodes['degree'], name='Normal', opacity=0.7, marker_color='blue'))
    degree_fig.add_trace(go.Histogram(x=anomaly_nodes['degree'], name='Anomaly', opacity=0.7, marker_color='red'))
    degree_fig.update_layout(title='Node Degree Distribution', xaxis_title='Degree', yaxis_title='Count', barmode='overlay')
    
    return html.Div([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=cpu_fig)], width=4),
            dbc.Col([dcc.Graph(figure=mem_fig)], width=4),
            dbc.Col([dcc.Graph(figure=degree_fig)], width=4),
        ])
    ])

def create_comparison_tab(dbscan_eps):
    # Run DBSCAN for comparison
    try:
        features_df = data['nodes_df'][['cpu', 'mem', 'degree']].copy()
        dbscan_results = baseline.run_dbscan_with_params(
            features_df,
            data['nodes_df']['is_anomaly'],
            eps=dbscan_eps,
            min_samples=config.DBSCAN_MIN_SAMPLES
        )
        
        # GNN metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = data['predictions']['is_anomaly']
        y_pred_gnn = data['predictions']['gnn_prediction']
        
        gnn_metrics = {
            'precision': precision_score(y_true, y_pred_gnn, zero_division=0),
            'recall': recall_score(y_true, y_pred_gnn, zero_division=0),
            'f1': f1_score(y_true, y_pred_gnn, zero_division=0)
        }
        
        # Comparison chart
        comparison_df = pd.DataFrame({
            'Model': ['GNN', 'DBSCAN'],
            'Precision': [gnn_metrics['precision'], dbscan_results['precision']],
            'Recall': [gnn_metrics['recall'], dbscan_results['recall']],
            'F1-Score': [gnn_metrics['f1'], dbscan_results['f1']]
        })
        
        comparison_fig = px.bar(
            comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Metric', y='Score', color='Model', barmode='group',
            title='Model Performance Comparison'
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("GNN Model"),
                    html.P(f"Precision: {gnn_metrics['precision']:.3f}"),
                    html.P(f"Recall: {gnn_metrics['recall']:.3f}"),
                    html.P(f"F1-Score: {gnn_metrics['f1']:.3f}")
                ], width=6),
                dbc.Col([
                    html.H4("DBSCAN Model"),
                    html.P(f"Precision: {dbscan_results['precision']:.3f}"),
                    html.P(f"Recall: {dbscan_results['recall']:.3f}"),
                    html.P(f"F1-Score: {dbscan_results['f1']:.3f}")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=comparison_fig)], width=12)
            ])
        ])
    
    except Exception as e:
        return html.Div(f"Error running comparison: {e}")

def create_data_tab():
    return html.Div([
        html.H4("Dataset Overview"),
        html.P(f"Total Nodes: {len(data['nodes_df'])}"),
        html.P(f"Anomalous Nodes: {data['nodes_df']['is_anomaly'].sum()}"),
        html.P(f"Training Edges: {len(data['edges_train'])}"),
        html.P(f"Test Edges: {len(data['edges_test'])}"),
        
        html.H5("Sample Predictions"),
        html.Div([
            # Convert dataframe to table
            dbc.Table.from_dataframe(
                data['predictions'].head(10), 
                striped=True, 
                bordered=True, 
                hover=True
            )
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)