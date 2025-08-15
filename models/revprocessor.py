import torch
import torch.nn as nn
from torch_geometric.nn import GroupAddRev, GATConv, GENConv, GINEConv, MLP

class RevProcessor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        assert {'mp_type', 'num_mp_steps','num_groups', 'latent_dim', 'dropout'}.issubset(kwargs)

        # initialize the message-passing blocks
        self.mp_layers = nn.ModuleList()
        for i in range(kwargs['num_mp_steps']):
            mp_layer = ProcessorBlock(kwargs['latent_dim'] // kwargs['num_groups'], kwargs['latent_dim']// kwargs['num_groups'],
                                      kwargs['mp_type'])
            self.mp_layers.append(GroupAddRev(mp_layer, num_groups=kwargs['num_groups']))

        self.dropout = kwargs['dropout']

    def forward(self, x, edge_index, edge_attr, batch):
        # Generate a dropout mask which will be shared across the message-passing blocks:
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        # update the node latent features through the message-passing
        for layer in self.mp_layers:
            x = layer(x, edge_index, edge_attr, mask)

        return x

class ProcessorBlock(nn.Module):
    def __init__(self, node_latent_dim, edge_latent_dim, mp_type='GAT'):
        super().__init__()
        assert (mp_type in ['GAT', 'GEN', 'GINE'])

        if mp_type == 'GAT':
            self.conv = GATConv(in_channels=node_latent_dim, out_channels=node_latent_dim, edge_dim=edge_latent_dim, heads=2, add_self_loops=True, concat=False)
        elif mp_type == 'GEN':
            self.conv = GENConv(in_channels=node_latent_dim, out_channels=node_latent_dim, norm='layer', aggr='mean', num_layers=2)
        elif mp_type == 'GINE':
            gin_mlp =   MLP(in_channels = node_latent_dim, hidden_channels = node_latent_dim*2,out_channels = node_latent_dim,
                            num_layers = 2, act='relu', norm='LayerNorm')
            self.conv = GINEConv(gin_mlp, node_latent_dim, node_latent_dim, edge_dim=edge_latent_dim)
        else:
            raise NotImplementedError

        if mp_type == 'GAT':  # apply layer norm to the GAT only (GIN and GEN already have)
            self.norm = nn.LayerNorm(node_latent_dim, elementwise_affine=True)
        else:
            self.norm = nn.Identity()

    def reset_parameters(self):
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr, dropout_mask=None):
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.norm(self.conv(x, edge_index, edge_attr))
