import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from models import FRGT
from datatools import CFDGraphsDataset
from box import Box
import yaml
import os
from tqdm import tqdm
import string
import random

# Set multiprocessing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

def train(config_path: str):
    """Train the FRGT model with the specified configuration.
    
    This function handles the complete training pipeline including:
    - Dataset loading and preprocessing
    - Model initialization and configuration
    - Training loop with validation
    - Model checkpointing and saving
    
    Args:
        config_path (str): Path to the YAML configuration file containing
            training hyperparameters, model settings, and I/O configurations.
            
    """
    # Load configuration from YAML file
    # Box provides convenient dot notation access to nested config parameters
    config = Box.from_yaml(filename=parser.parse_args().config, Loader=yaml.FullLoader)

    # Initialize datasets and data loaders
    # Training dataset with augmentation options enabled
    train_dataset = CFDGraphsDataset(
        zip_path=config.io_settings.train_dataset_path,
        random_masking=config.hyperparameters.random_masking,  # Random fluid node masking for robustness
        rd_in_polar_coords=config.hyperparameters.rd_in_polar_coords,  # Polar coordinate edge features
        zero_augmentation=config.hyperparameters.zero_augmentation,  # Zero augmentation for sparse data
        airfoil_coverage=config.hyperparameters.airfoil_coverage  # Sensor coverage fraction (0-1)
    )
    # Training data loader with graph batching
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.hyperparameters.batch_size, 
        shuffle=True,
        exclude_keys=['triangles', 'triangle_points'],  # Exclude mesh triangulation data from train set for memory efficiency
        num_workers=config.run_settings.num_t_workers, 
        pin_memory=False,
        persistent_workers=False if config.run_settings.num_t_workers == 0 else True
    )
    # Initialize validation dataset if validation is enabled
    if config.run_settings.validate:
        # Validation dataset without augmentation for consistent evaluation
        validate_dataset = CFDGraphsDataset(
            zip_path=config.io_settings.valid_dataset_path,
            random_masking=False,  # No random masking for deterministic validation
            rd_in_polar_coords=config.hyperparameters.rd_in_polar_coords,
            zero_augmentation=False,  # No augmentation for validation
            airfoil_coverage=config.hyperparameters.airfoil_coverage
        )
        # Validation loader with batch size 1 for individual graph evaluation
        validate_loader = DataLoader(
            validate_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=config.run_settings.num_v_workers, 
            pin_memory=False,
            persistent_workers=False if config.run_settings.num_v_workers == 0 else True
        )

    # Extract data dimensions from dataset and add to configuration
    # This provides model with input/output dimensions automatically
    config.data_dims = train_dataset.get_data_dims_dict()

    # Generate unique run identifier for this training session
    uid = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = 'proc_{}_dim_{}_uid_{}'.format(
        config.model_settings.processor_type, 
        config.model_settings.latent_dim, 
        uid
    )

    # Create directories for saving models and configurations
    # Each run gets its own directory for reproducibility
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'))
    # Save configuration file to run directory for future reference
    config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))

    # Set device for training (GPU if available, CPU otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize FRGT model with configuration parameters
    # Combines data dimensions, hyperparameters, and model settings
    model = FRGT(**config.data_dims, **config.hyperparameters, **config.model_settings)

    # Load pretrained model weights if specified
    if config.io_settings.pretrained_model:
        checkpoint = torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded pretrained model from {config.io_settings.pretrained_model}')

    # Move model to the appropriate device (GPU/CPU)
    model.to(device)

    # Initialize AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config.hyperparameters.start_lr), 
        weight_decay=float(config.hyperparameters.weight_decay)
    )

    # Cosine annealing learning rate scheduler for smooth decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.hyperparameters.epochs
    )

    # Optional: Enable anomaly detection to catch NaN gradients during debugging
    # torch.autograd.set_detect_anomaly(True)

    # Begin training loop
    print('Starting run {} on {}'.format(run_name, next(model.parameters()).device))
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Initialize progress bar for training epochs
    pbar = tqdm(total=config.hyperparameters.epochs)
    pbar.set_description('Training')
    # Main training loop over epochs
    for epoch in range(config.hyperparameters.epochs):
        # Initialize loss accumulators for this epoch
        train_loss = 0
        train_node_loss = 0
        train_div_loss = 0
        
        # Set model to training mode (enables dropout, batch norm, etc.)
        model.train()

        # Mini-batch training loop
        for i_batch, data in enumerate(train_loader):
            # Move batch data to the appropriate device (GPU/CPU)
            data = data.to(device)

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass: compute predictions and loss components
            data = model(data)
            batch_loss, batch_node_loss, batch_div_loss = model.compute_loss(data)
            
            # Accumulate losses for epoch statistics
            train_loss += batch_loss.item()
            train_node_loss += batch_node_loss.item()
            train_div_loss += batch_div_loss.item()

            # Backward pass: compute gradients and update parameters
            batch_loss.backward()
            optimizer.step()

        # Compute average training losses for this epoch
        train_loss = train_loss / len(train_loader)
        train_node_loss = train_node_loss / len(train_loader)
        train_div_loss = train_div_loss / len(train_loader)

        # Update learning rate according to cosine annealing schedule
        scheduler.step()

        # Save model checkpoint at specified intervals
        if (epoch + 1) % config.io_settings.save_epochs == 0:
            torch.save(
                {'model_state_dict': model.state_dict()},
                os.path.join(current_run_dir, 'trained_models', 'e{}.pt'.format(epoch + 1))
            )


        # Validation phase (if enabled)
        if config.run_settings.validate:
            # Initialize validation loss accumulators
            validation_loss = 0
            validation_loss_denormed = 0  # Loss computed on denormalized data
            validation_node_loss = 0
            validation_div_loss = 0

            # Set model to evaluation mode (disables dropout, etc.)
            model.eval()
            # Disable gradient computation for validation (memory efficiency)
            with torch.no_grad():
                for i_batch, data in enumerate(validate_loader):
                    # Move validation data to device
                    data = data.to(device)

                    # Forward pass without gradient computation
                    data = model(data)
                    
                    # Compute validation loss metrics on normalized data
                    batch_loss, batch_node_loss, batch_div_loss = model.compute_loss(data)
                    validation_loss += batch_loss
                    validation_node_loss += batch_node_loss
                    validation_div_loss += batch_div_loss
                    
                    # Denormalize predictions and targets for physical interpretation
                    data.x = data.x * data.node_norm_vals[:, 1] + data.node_norm_vals[:, 0]
                    data.y = data.y * data.node_norm_vals[:, 1] + data.node_norm_vals[:, 0]
                    validation_loss_denormed += model.compute_loss(data)[0]
                    
            # Compute average validation losses across all validation samples
            validation_loss = validation_loss / len(validate_loader)
            validation_loss_denormed = validation_loss_denormed / len(validate_loader)
            validation_node_loss = validation_node_loss / len(validate_loader)
            validation_div_loss = validation_div_loss / len(validate_loader)
  
            # Update progress bar with training and validation losses
            pbar.set_postfix({
                'Train Loss': f'{train_loss:.8f}',
                'Validation Loss': f'{validation_loss:.8f}'
            })
            pbar.update(1)

            # Save best model based on validation loss (early stopping criterion)
            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(
                        {'model_state_dict': model.state_dict()},
                        os.path.join(current_run_dir, 'trained_models', 'best.pt')
                    )

        else:
            # Update progress bar with training loss only (no validation)
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}'})
            pbar.update(1)
            
if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Train FRGT model for flow reconstruction from sparse measurements'
    )
    parser.add_argument(
        '--config', '-c', 
        help='Path to the YAML configuration file', 
        type=str, 
        required=True
    )
    
    # Parse arguments and start training
    args = parser.parse_args()
    train(config_path=args.config)