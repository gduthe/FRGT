import torch.nn as nn
from torch_geometric.nn import MLP

class FRDecoder(nn.Module):
    """Feature decoder for FRGT that maps latent representations to physical quantities.
    
    The decoder transforms high-dimensional latent node features back to the physical
    space, producing the final flow field predictions (pressure, velocity components).
    The decoder uses a plain last layer (no activation/normalization) to allow 
    unrestricted output values.
    
    Args:
        latent_dim (int): Dimension of the latent space
        num_dec_mlp_layers (int): Number of MLP layers in the decoder
        node_out_dim (int): Output node feature dimension (typically 3 for pressure, u, v)
        dropout (float): Dropout probability for regularization
    """
    
    def __init__(self, **kwargs):
        """Initialize the FRGT decoder with the specified configuration."""
        super().__init__()

        assert {'latent_dim', 'num_dec_mlp_layers', 'node_out_dim', 'dropout'}.issubset(kwargs)

        # Initialize node decoder MLP with plain last layer for unrestricted outputs
        # Maps latent_space -> [pressure, u_velocity, v_velocity]
        self.node_decoder = MLP(
            in_channels=kwargs['latent_dim'], 
            out_channels=kwargs['node_out_dim'],
            hidden_channels=kwargs['latent_dim'], 
            num_layers=kwargs['num_dec_mlp_layers'],
            act='relu', 
            norm='layer', 
            dropout=kwargs['dropout'], 
            plain_last=True  # No activation/normalization on final layer
        )

    def forward(self, x):
        # Decode latent features back to physical space
        x = self.node_decoder(x)

        return x