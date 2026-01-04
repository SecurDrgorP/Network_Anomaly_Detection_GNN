import torch
from torch_geometric.nn import SAGEConv, GAE

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

def build_model(in_dim):
    import config
    encoder = GraphSAGEEncoder(in_dim, config.HIDDEN_DIM, config.LATENT_DIM, config.DROPOUT)
    return GAE(encoder)
