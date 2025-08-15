from .encoder import FREncoder
from .revprocessor import RevProcessor
from .hgtprocessor import HGTProcessor, HGTProcessorIT
from .decoder import FRDecoder
from .feature_propagation import FeaturePropagator
import torch.nn as nn
import torch
from torch_scatter import scatter_sum

class FRGT(nn.Module):
    """Flow Reconstructing Graph Transformer.
    
    A deep learning model for reconstructing complete fluid flow fields from sparse
    sensor measurements using graph neural networks and transformer architectures.
    
    Args:
        processor_type (str): Type of processor to use ('rev', 'hgt', 'hgt_it')
        encoder_settings (dict): Configuration for the encoder module
        processor_settings (dict): Configuration for the processor module  
        decoder_settings (dict): Configuration for the decoder module
        fp_steps (int): Number of feature propagation iterations
        div_loss_factor (float): Weight for divergence loss component
        noise_sigma (float): Standard deviation of noise added during training
        **kwargs: Additional model configuration parameters
    """
    def __init__(self, **kwargs):
        super().__init__()
        assert {'processor_type', 'encoder_settings', 'processor_settings', 'decoder_settings', 'fp_steps',
                 'div_loss_factor', 'noise_sigma'}.issubset(kwargs)
        
        # Initialize the feature propagator for interpolating missing measurements
        self.feature_propagator = FeaturePropagator(num_iterations=kwargs['fp_steps'])

        # Initialize the encoder to map input features to latent space
        self.encoder = FREncoder(**kwargs, **kwargs['encoder_settings'])
        
        # Initialize the appropriate processor based on configuration
        self.processor_type = kwargs['processor_type']
        if self.processor_type == 'rev':
            # Reversible GNN processor for memory efficiency
            self.processor = RevProcessor(**kwargs, **kwargs['processor_settings'])
        elif self.processor_type == 'hgt':
            # Hybrid Graph Transformer processor
            self.processor = HGTProcessor(**kwargs, **kwargs['processor_settings'])
        elif self.processor_type == 'hgt_it':
            # Interleaved Hybrid Graph Transformer processor
            self.processor = HGTProcessorIT(**kwargs, **kwargs['processor_settings'])
        else:
            raise NotImplementedError(f'Processor type {self.processor_type} is not implemented')

        # Initialize the decoder to map latent features back to physical space
        self.decoder = FRDecoder(**kwargs, **kwargs['decoder_settings'])

        # Store physics-informed loss weighting factor
        self.div_loss_factor = kwargs['div_loss_factor']

        # Store noise level for training regularization
        self.noise_sigma = kwargs['noise_sigma']
        
    def forward(self, data):
        assert all(hasattr(data, attr) for attr in ['x', 'known_feature_mask', 'edge_index', 'edge_attr', 'globals', 'batch'])
        
        # Step 1: Propagate missing features using graph connectivity to initialioze input to the encoder
        # This interpolates unknown measurements from known neighboring values
        filled_features = self.feature_propagator.propagate(x=data.x.clone(), edge_index=data.edge_index, mask=data.known_feature_mask)
        data.x = torch.where(data.known_feature_mask, data.x, filled_features)

        # Step 2: Encode input features to latent space
        # Maps physical quantities and geometry to high-dimensional representations
        data.x, data.edge_attr = self.encoder(data.x, data.edge_attr, data.globals, data.batch)
        
        # Step 3: Apply noise regularization during training for robustness
        if self.training and self.noise_sigma != 0.0:
            noise = torch.randn_like(data.x) * self.noise_sigma
            # Gradient trick: allow gradients to flow through as if no noise was added
            data.x = ((noise + data.x).detach() - data.x).detach() + data.x

        # Step 4: Process the encoded graph with message passing and attention
        # This captures long-range dependencies and fluid dynamics relationships
        data.x = self.processor(data.x, data.edge_index, data.edge_attr, data.batch)
            
        # Step 5: Decode latent features back to physical space
        # Maps high-dimensional representations to [pressure, u_velocity, v_velocity]
        data.x = self.decoder(data.x)

        return data
        
    def compute_loss(self, data):
        """Compute the total training loss combining reconstruction and physics constraints.
        
        The loss function combines:
        1. **Node reconstruction loss**: MSE between predicted and target node features
        2. **Divergence loss**: Physics-informed constraint enforcing incompressible flow
        
        Args:
            data (torch_geometric.data.Data): Graph data with predictions in data.x and 
                targets in data.y
                
        Returns:
            tuple: (total_loss, node_loss, divergence_loss)
                - total_loss (torch.Tensor): Combined weighted loss
                - node_loss (torch.Tensor): MSE reconstruction loss
                - divergence_loss (torch.Tensor): Physics constraint loss
        """
        loss_fn = nn.MSELoss()
        
        # Compute primary reconstruction loss between predictions and targets
        node_loss = loss_fn(data.x, data.y)

        # Add physics-informed divergence constraint if enabled
        if self.div_loss_factor != 0.0:
            div_loss = self.compute_div_loss(data)
        else:
             div_loss = torch.tensor(0.0, device=data.x.device)

        # Combine losses with weighting factor
        overall_loss = node_loss + self.div_loss_factor * div_loss

        return overall_loss, node_loss, div_loss

    def compute_div_loss(self, data):
        """Compute divergence loss to enforce incompressible flow physics.
        
        This implements the incompressible flow constraint ∇·u = 0 by:
        1. Computing velocity flux through each edge
        2. Summing fluxes for each node (discrete divergence)
        3. Penalizing non-zero divergence in fluid regions
        
        The discrete divergence is computed as: div(u) ≈ Σ(u·n·A) where u is velocity,
        n is the edge normal, and A is the edge face area.
        
        Args:
            data (torch_geometric.data.Data): Graph data containing:
                - x: Node features with velocity components [pressure, u, v, ...]
                - edge_index: Graph connectivity
                - edge_rd: Relative distance vectors between nodes
                - edge_s: Edge face surface areas
                - node_type: Node type classification (0=fluid, 1=boundary)
                
        Returns:
            torch.Tensor: Mean squared divergence loss for fluid nodes
        """
        # Extract velocity vectors [u, v] for source nodes of each edge
        u = data.x[data.edge_index[0, :], 1:3]
        
        # Compute flux through each edge: j = (u · edge_rd) * edge_surface
        # This approximates the dot product u·n·A where n is edge normal and A is area
        j = torch.bmm(u.view(data.edge_index.shape[1], 1, -1), 
                      data.edge_rd.view(data.edge_index.shape[1], -1, 1))
        j = j.squeeze() * data.edge_s.squeeze()
        
        # Sum all outgoing fluxes for each node (discrete divergence operator)
        div = scatter_sum(j, data.edge_index[0, :], dim=0)

        # Apply divergence constraint only to fluid nodes (not boundary nodes)
        div = div[data.node_type == 0]

        # Penalize non-zero divergence (incompressible flow constraint)
        return nn.functional.mse_loss(div, torch.zeros_like(div))