import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool

# ------------------------ MPNN LAYER ------------------------

class MPNNLayer(MessagePassing):
    '''Message Passing Layer.'''
    
    def __init__(self, node_features, edge_features, hidden_features, out_features, aggr, act):
        '''Initialize the MPNNLayer.

        Parameters:
        - node_features (int): Number of node features.
        - edge_features (int): Number of edge features.
        - hidden_features (int): Number of hidden features.
        - out_features (int): Number of output features.
        - aggr (str): Aggregation method. Can be 'add', 'mean' or 'max'.
        - act (torch.nn.Module): Activation function.
        '''
        super().__init__(aggr=aggr)

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, edge_index, edge_attr=None):
        '''Forward pass of the MPNNLayer.'''
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        '''Construct messages between nodes.'''
        input = [x_i, x_j] if edge_attr is None else [x_i, x_j, edge_attr]
        input = torch.cat(input, dim=-1)
        return self.message_net(input)

    def update(self, message, x):
        '''Update node features.'''
        input = torch.cat((x, message), dim=-1)
        return self.update_net(input)

# ------------------------ MPNN ------------------------

class MPNN(nn.Module):
    '''Message Passing Neural Network.'''
    
    def __init__(self, node_features, edge_features, hidden_features, out_features, 
                 num_layers, aggr, act, pool=None):
        '''Initialize the MPNN.

        Parameters:
        - node_features (int): Number of node features.
        - edge_features (int): Number of edge features.
        - hidden_features (int): Number of hidden features.
        - out_features (int): Number of output features.
        - num_layers (int): Number of MPNN layers.
        - aggr (str): Aggregation method. Can be 'add', 'mean' or 'max'.
        - act (torch.nn.Module): Activation function.
        - pool (str, optional): Pooling method. Can be 'add', 'mean' or None.
        '''
        super().__init__()

        self.embedder = nn.Sequential(
            nn.Linear(node_features, hidden_features),
            act(),
            nn.Linear(hidden_features, hidden_features)
        )
        layers = [
            MPNNLayer(
                node_features=hidden_features,
                hidden_features=hidden_features,
                edge_features=edge_features,
                out_features=hidden_features,
                aggr=aggr,
                act=act
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

        if pool is None:
            self.pooler = None
        elif pool == "add":
            self.pooler = global_add_pool
        elif pool == "mean":
            self.pooler = global_mean_pool
        else:
            raise Exception("Pooler not recognized")

        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        '''Forward pass of the MPNN.'''
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        if self.pooler:
            x = self.pooler(x, batch)
        return self.head(x)

    def get_pre_pool_rep(self, x, edge_index, edge_attr=None):
        '''Get the representation before pooling.'''
        with torch.no_grad():
            x = self.embedder(x)
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
        return x