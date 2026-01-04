import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def create_pyg_data(df_nodes, df_edges):
    features = df_nodes[['cpu', 'mem']].values
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(features), dtype=torch.float)
    edge_index = torch.tensor([df_edges['src'].values, df_edges['dst'].values], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)
