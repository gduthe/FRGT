# FRGT: Flow Reconstructing Graph Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)

A deep learning framework for reconstructing fluid flow fields from sparse measurements using Graph Transformer (and GNN) architectures. FRGT combines the strengths of graph-based representations with attention mechanisms to predict complete flow fields from sparse observations.

![FRGT Architecture](gt_archi.png)

## Overview

FRGT (Flow Reconstructing Graph Transformer) is designed to reconstruct complete fluid flow fields from sparse sensor measurements on 2D airfoils. The model leverages:

- **Graph Neural Networks** for handling irregular mesh geometries
- **Transformer attention** for long-range dependencies
- **Multi-processor architectures** (Reversible GNN, Hybrid Graph Transformer)

## Key Features

- ⚡ **Efficient reconstruction** from sparse pressure measurements
- 🔄 **Multiple processor types**: HGT, Reversible GNN, Interleaved HGT
- 📐 **Optional physics-aware training** using the divergence loss
- 🎯 **Configurable coverage** for airfoil measurements (0-100%)

## Project Structure

```
FRGT/
├── README.md                    # Project documentation
├── LICENSE.md                   # MIT License
├── requirements.txt             # Python dependencies
├── train.py                     # Main training script
├── gt_archi.png                # Architecture diagram
├── configs/                     # Configuration files
│   ├── config_frgt.yml         # Standard FRGT config
│   ├── config_frgt_it.yml      # Interleaved transformer config
│   └── config_revGAT.yml       # Reversible GNN config
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── frgt.py                 # Main FRGT model
│   ├── encoder.py              # Feature encoder
│   ├── decoder.py              # Output decoder
│   ├── hgtprocessor.py         # Hybrid Graph Transformer
│   ├── revprocessor.py         # Reversible GNN processor
│   ├── feature_propagation.py  # Feature propagation utilities
│   └── building_blocks.py      # Neural network components
├── datatools/                   # Data handling utilities
│   ├── __init__.py
│   ├── dataset.py              # CFD dataset loader
│   ├── parse_mesh.py           # Mesh parsing utilities
│   ├── process_simulations.py  # Simulation preprocessing
│   └── compute_dataset_stats.py # Dataset statistics
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── single_model_plotter.py # Visualization tools
├── notebooks/                   # Jupyter notebooks
│   └── results_viz.ipynb       # Results visualization
└── runs/                        # Training outputs
    └── proc_*/                  # Individual training runs
        ├── config.yml          # Run configuration
        └── trained_models/     # Model checkpoints
            ├── best.pt         # Best validation model
            └── e*.pt           # Epoch checkpoints
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large datasets

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gduthe/FRGT.git
   cd FRGT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import torch_geometric; print('Installation successful!')"
   ```

## Dataset Requirements

FRGT expects CFD simulation data in a specific format:

### Data Format
- **Input**: ZIP file containing PyTorch geometric graphs (.pt files)
- **Node features**: Pressure, velocities (u,v), signed distance function (SDF)
- **Edge features**: Relative distances, face surfaces
- **Global features**: Estimated free-stream velocity

### Dataset Structure
Each graph contains:
- `x`: Node features [pressure, sdf] (input)
- `y`: Target features [pressure, u_velocity, v_velocity]
- `edge_index`: Graph connectivity
- `edge_attr`: Edge features [dx, dy, face_surface]
- `pos`: Node coordinates
- `node_type`: Node classification (0=fluid, 1=boundary)

### Obtaining Data

For this research, CFD simulations were generated using OpenFOAM with:
- **Geometry**:  airfoils from the [UIUC dataset](https://m-selig.ae.illinois.edu/ads/coord_database.html) with varying angles of attack
- **Solver**: Incompressible, steady-state k-omega SST 
- **Mesh**: Unstructured triangular meshes
- **Boundary conditions**: Far-field velocity inlet, pressure outlet

The training dataset can be obtained from [Zenodo](https://zenodo.org/records/14629208). If you have OpenFOAM simulations, you can use the preprocessing pipeline in `datatools/` to create custom datasets.

## Usage

### Basic Training

1. **Prepare your dataset** and update paths in the config file:
   ```yaml
   io_settings:
     train_dataset_path: 'path/to/train_dataset.zip'
     valid_dataset_path: 'path/to/valid_dataset.zip'
   ```

2. **Run training:**
   ```bash
   python train.py --config configs/config_frgt.yml
   ```

### Configuration Options

#### Model Architectures
- **HGT**: `processor_type: 'hgt'` - Hybrid Graph Transformer
- **Reversible GNN**: `processor_type: 'rev'` - Memory-efficient reversible layers  
- **Interleaved HGT**: `processor_type: 'hgt_it'` - Interleaved attention layers

#### Key Parameters
```yaml
hyperparameters:
  batch_size: 1                 # Graphs per batch
  epochs: 500                   # Training epochs
  start_lr: 5e-4               # Initial learning rate
  airfoil_coverage: 1.0        # Sensor coverage (0.0-1.0)
  
model_settings:
  latent_dim: 160              # Hidden dimension
  fp_steps: 30                 # Feature propagation steps
  processor_type: 'hgt'        # Architecture type
```

### Advanced Usage

#### Custom Airfoil Coverage
```bash
# Train with 70% airfoil sensor coverage
python train.py --config configs/config_frgt.yml
# Modify airfoil_coverage: 0.7 in config
```

#### Physics-Informed Training
```yaml
hyperparameters:
  div_loss_factor: 0.1  # Enable divergence loss
```

#### Noise Robustness
```yaml
model_settings:
  noise_sigma: 0.01  # Add noise during training
```

## Model Outputs

Training produces:
- **Model checkpoints**: `runs/proc_*/trained_models/`
- **Best model**: `best.pt` (lowest validation loss)
- **Configuration**: `config.yml` (run parameters)
- **Periodic saves**: `e*.pt` (every N epochs)

## Evaluation

Use the provided visualization notebook:
```bash
jupyter notebook notebooks/results_viz.ipynb
```

Or implement custom evaluation:
```python
import torch
from models import FRGT
from datatools import CFDGraphsDataset

# Load trained model
model = FRGT(**config)
checkpoint = torch.load('runs/proc_*/trained_models/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test data
test_dataset = CFDGraphsDataset('path/to/test_dataset.zip')
# ... evaluation code
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{duthe2025graph,
  title={Graph Transformers for inverse physics: reconstructing flows around arbitrary 2D airfoils},
  author={Duth{\'e}, Gregory and Abdallah, Imad and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2501.17081},
  year={2025}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

