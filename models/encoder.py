import torch.nn as nn
import torch
from torch_geometric.nn import MLP

class FREncoder(nn.Module):
    """Feature encoder for FRGT that maps input features to latent space.
    
    The encoder transforms raw node and edge features into high-dimensional latent
    representations that capture the essential information for flow reconstruction.
    It supports optional global feature concatenation for enhanced context.
    
    Args:
        use_est_globals (bool): Whether to concatenate global features to node features
        latent_dim (int): Dimension of the latent space
        num_enc_mlp_layers (int): Number of MLP layers in the encoder
        node_feature_dim (int): Input node feature dimension
        edge_feature_dim (int): Input edge feature dimension
        glob_feature_dim (int): Global feature dimension
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(self, **kwargs):
        """Initialize the FRGT encoder with the specified configuration."""
        super().__init__()

        assert {'use_est_globals', 'latent_dim', 'num_enc_mlp_layers', 'node_feature_dim',
                 'edge_feature_dim','glob_feature_dim', 'dropout'}.issubset(kwargs)
        
        # Calculate input dimension for node encoder
        if kwargs['use_est_globals']:
            # Concatenate global features to each node
            node_input_dim = kwargs['node_feature_dim'] + kwargs['glob_feature_dim']
        else:
            # Use only node features
            node_input_dim = kwargs['node_feature_dim']

        # Initialize multi-layer perceptrons for feature encoding
        # Node encoder: maps [node_features, (global_features)] -> latent_space
        self.node_encoder = MLP(
            in_channels=node_input_dim, 
            hidden_channels=kwargs['latent_dim'], 
            out_channels=kwargs['latent_dim'], 
            num_layers=kwargs['num_enc_mlp_layers'], 
            act='relu', 
            norm='layer', 
            dropout=kwargs['dropout']
        )
        
        # Edge encoder: maps edge_features -> latent_space
        self.edge_encoder = MLP(
            in_channels=kwargs['edge_feature_dim'], 
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['latent_dim'], 
            num_layers=kwargs['num_enc_mlp_layers'], 
            act='relu', 
            norm='layer', 
            dropout=kwargs['dropout']
        )
        
        self.use_est_globals = kwargs['use_est_globals']
        
    def forward(self, x, edge_attr, globals_est, batch):
        if self.use_est_globals:
            # Concatenate global features to each node based on batch assignment
            x_enc = self.node_encoder(torch.cat([x, globals_est[batch]], dim=1))
        else:
            # Encode nodes without global information
            x_enc = self.node_encoder(x)
            
        # Encode edge features independently
        edge_attr_enc = self.edge_encoder(edge_attr)

        return x_enc, edge_attr_enc