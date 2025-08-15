from torch_geometric.data import Dataset
import torch_geometric.transforms as T
import torch
from zipfile import ZipFile
import io
import numpy as np

class CFDGraphsDataset(Dataset):
    """Dataset comprised of 2D OpenFOAM CFD simulations for flow reconstruction.
    
    This dataset loads preprocessed CFD simulation graphs from a ZIP archive, with each
    simulation representing flow around a 2D airfoil. The dataset supports various
    augmentation and configuration options for training flow reconstruction models.
    
    Args:
        zip_path (str): Path to ZIP file containing preprocessed graph data (.pt files)
        sdf_input (bool, optional): Include signed distance function as input feature. 
            Defaults to True.
        rd_in_polar_coords (bool, optional): Convert edge relative distances to polar 
            coordinates. Defaults to False.
        random_masking (bool, optional): Apply random masking to fluid nodes during 
            training. Defaults to False.
        zero_augmentation (bool, optional): Apply 5% chance of zero augmentation. 
            Defaults to False.
        airfoil_coverage (float, optional): Fraction of airfoil chord with sensor 
            measurements (0.0-1.0). Defaults to 1.0.
        transform (callable, optional): PyTorch Geometric transform to apply.
        pre_transform (callable, optional): PyTorch Geometric pre-transform to apply.
    """

    def __init__(self, zip_path: str, sdf_input:bool=True, rd_in_polar_coords:bool=False, random_masking: bool=False,
                 zero_augmentation:bool=False, airfoil_coverage=1, transform=None, pre_transform=None):
        assert 0 < airfoil_coverage <= 1, "airfoil_coverage must be between 0 and 1"
        super().__init__(None, transform, pre_transform)
        
        # Store dataset path and count number of simulations
        self.__zip_path = zip_path
        with ZipFile(zip_path, 'r') as zf:
            self.__num_graphs = len(zf.namelist())

        # Store configuration options
        self.sdf_input = sdf_input
        self.random_masking = random_masking
        self.rd_in_polar_coords = rd_in_polar_coords
        self.zero_augmentation = zero_augmentation
        self.airfoil_coverage = airfoil_coverage

    def __getitem__(self, idx):
        """Load and preprocess a single CFD simulation graph."""
        
        # Load graph data from ZIP archive
        with ZipFile(self.__zip_path, 'r') as zf:
            with zf.open(zf.namelist()[idx]) as item:
                stream = io.BytesIO(item.read())
                data = torch.load(stream, weights_only=False)
        
        # Ensure all features are float tensors for computation
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.pos = data.pos.float()
                
        # Extract edge geometry for physics-informed loss computation
        # edge_rd: relative distance vectors between connected nodes
        # edge_s: face surface areas for flux calculations
        data.edge_rd = data.edge_attr[:, 0:2]
        data.edge_s = data.edge_attr[:, -1:]

        # Convert edge features to polar coordinates if enabled
        if self.rd_in_polar_coords:
            polar_transf = T.Polar(norm=False, cat=True)
            data = polar_transf(data)
            data.edge_attr = data.edge_attr[:, [-1, -2, -3]].float()
            data.edge_attr_labels = ['theta', 'edge_length', 'face_surface']

        # Apply zero augmentation for robustness training (5% chance)
        if self.zero_augmentation:
            augment = np.random.choice([False, True], 1, p=[0.95, 0.05])
            if augment:
                data.x = torch.zeros_like(data.x)
        
        # Extract pressure measurements from airfoil sensors based on coverage
        # airfoil_coverage controls what fraction of chord has sensors 
        if self.airfoil_coverage < 1:
            airf_nodes = data.node_type == 1.0
            # Measured nodes: airfoil nodes within coverage range (chord length = 1)
            measured_airf_nodes = airf_nodes * (data.pos[:, 0] < self.airfoil_coverage)
            unmeasured_airf_nodes = airf_nodes * (data.pos[:, 0] >= self.airfoil_coverage)
            airfoil_pressure = data.x[measured_airf_nodes, 0]
        else:
            # Full coverage: use all airfoil node pressures
            airfoil_pressure = data.x[data.node_type == 1.0, 0]
        
        # Compute normalization statistics from airfoil pressure measurements
        airfoil_pressure_mean = airfoil_pressure.mean(dim=0)
        airfoil_pressure_std = airfoil_pressure.std(dim=0)
        
        # Estimate freestream velocity using Bernoulli's equation
        # Assumes incompressible flow: U_inf = sqrt(2*P_max/rho)
        # where rho = 1.225 kg/m³ (air density at sea level)
        Uinf_est = torch.sqrt(2*airfoil_pressure.max()/1.2250)
        data.globals = torch.tensor([Uinf_est]).unsqueeze(0)
        
        # Normalize node features for stable training
        # Pressure: zero-mean, unit variance based on airfoil measurements
        # Velocities: scaled by estimated freestream velocity
        data.x[:,0] = self.__normalize(data.x[:,0], subtract=airfoil_pressure_mean, divide=airfoil_pressure_std)
        data.x[:,1] = self.__normalize(data.x[:,1], subtract=0, divide=Uinf_est)
        data.x[:,2] = self.__normalize(data.x[:,2], subtract=0, divide=Uinf_est)
        # Store normalization parameters for denormalization during evaluation
        data.node_norm_vals = torch.tensor([[airfoil_pressure_mean, airfoil_pressure_std], [0, Uinf_est], [0, Uinf_est]])
        
        # Set target values: [pressure, u_velocity, v_velocity] (excluding SDF at index 3)
        data.y = data.x[:, :3].detach().clone()

        # Configure input features based on SDF setting
        if self.sdf_input:
            # Use pressure + signed distance function as input
            data.x = data.x[:, [0,3]]
            data.input_node_feat_labels = ['pressure', 'sdf']
        else:
            # Use only pressure as input (more challenging sparse reconstruction)
            data.x = data.x[:, 0:1]
            data.input_node_feat_labels = ['pressure']

        # Apply random masking to fluid nodes for data augmentation
        if self.random_masking:
            # Randomly keep 70-100% of fluid node measurements
            masked_fluid_proportion = np.random.uniform(0.7, 1.0)
        else:
            # Keep all fluid measurements
            masked_fluid_proportion = 1.0
        
        # Generate random mask and apply to fluid nodes only
        mask = torch.tensor(np.random.choice([0, 1], size=data.num_nodes, 
                           p=[1-masked_fluid_proportion, masked_fluid_proportion])) > 0
        # Mask pressure values in fluid region (node_type == 0)
        self.__mask_nodes(data.x[:, 0], mask * (data.node_type == 0.0), mask_value=float('nan')) 
        
        # Mask unmeasured airfoil nodes based on sensor coverage
        # This simulates sparse sensor placement on the airfoil surface
        if self.airfoil_coverage < 1:
            self.__mask_nodes(data.x[:, 0], unmeasured_airf_nodes, mask_value=float('nan'))
    
        # Add one-hot encoded node type to input features
        # This helps the model distinguish between fluid (0) and boundary (1) nodes
        data.x = torch.cat((data.x, torch.nn.functional.one_hot(data.node_type)), dim=1)

        # Create mask indicating which features are known (not NaN)
        # This is used by the feature propagation module
        data.known_feature_mask = ~torch.isnan(data.x)

        return data

    def __mask_nodes(self, x, mask_tensor, mask_value=float('nan')):
        """Apply masking to selected nodes by setting their values to mask_value.
        
        Args:
            x (torch.Tensor): Node features to mask
            mask_tensor (torch.Tensor): Boolean tensor indicating which nodes to mask
            mask_value (float): Value to set for masked nodes (default: NaN)
        """
        x[mask_tensor] = torch.tensor(mask_value)
        
    @property
    def num_glob_features(self) -> int:
        """Returns the number of global features in the dataset.
        
        Returns:
            int: Number of global features (typically 1 for freestream velocity)
        """
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals.shape[1]

    @property
    def num_glob_output_features(self) -> int:
        """Returns the number of global output features in the dataset.
        
        Returns:
            int: Number of global output features
        """
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.globals_y.shape[1]

    @property
    def num_node_output_features(self) -> int:
        """Returns the number of node output features in the dataset.
        
        Returns:
            int: Number of node output features (3 for pressure, u_velocity, v_velocity)
        """
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.y.shape[1]

    def get_data_dims_dict(self) -> dict:
        """Returns a dictionary with the number of features for each data type.
        
        Returns:
            dict: Dictionary containing:
                - node_feature_dim: Input node feature dimension
                - edge_feature_dim: Edge feature dimension  
                - glob_feature_dim: Global feature dimension
                - node_out_dim: Output node feature dimension
        """
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return {
            'node_feature_dim': data.x.shape[1], 
            'edge_feature_dim': data.edge_attr.shape[1], 
            'glob_feature_dim': data.globals.shape[1], 
            'node_out_dim': data.y.shape[1]
        }


    def __len__(self):
        """Returns the number of graphs in the dataset."""
        return self.__num_graphs
    
    def __normalize(self, x: torch.tensor, subtract=None, divide=None):
        """Normalize tensor features with optional custom normalization parameters.
        
        Args:
            x (torch.Tensor): Input tensor to normalize
            subtract (float, optional): Value to subtract (default: mean)
            divide (float, optional): Value to divide by (default: std)
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        if subtract is None:
            # Per feature normalization using mean
            subtract = x.mean(dim=0)
        if divide is None:
            # Per feature normalization using standard deviation
            divide = x.std(dim=0)

        # Handle zero division for standard deviation normalization
        divide[divide == 0.0] = 1.0

        return (x - subtract) / divide