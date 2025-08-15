"""
Feature Propagation Module from: 

Rossi, Emanuele, et al. "On the unreasonable effectiveness of feature propagation in learning on
graphs with missing node features." Learning on graphs conference. PMLR, 2022.
"""

from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add
import torch
from torch import Tensor


def get_adj_row_sum(edge_index, edge_weight, n_nodes):
    """
    Get weighted out degree for nodes. This is equivalent to computing the sum of the rows of the weighted adjacency matrix.
    """
    row = edge_index[0]
    return scatter_add(edge_weight, row, dim=0, dim_size=n_nodes)


def get_adj_col_sum(edge_index, edge_weight, n_nodes):
    """
    Get weighted in degree for nodes. This is equivalent to computing the sum of the columns of the weighted adjacency matrix.
    """
    col = edge_index[1]
    return scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)


def row_normalize(edge_index, edge_weight, n_nodes):
    row_sum = get_adj_row_sum(edge_index, edge_weight, n_nodes)
    row_idx = edge_index[0]
    return edge_weight / row_sum[row_idx]


def col_normalize(edge_index, edge_weight, n_nodes):
    col_sum = get_adj_col_sum(edge_index, edge_weight, n_nodes)
    col_idx = edge_index[1]
    return edge_weight / col_sum[col_idx]

def get_symmetrically_normalized_adjacency(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return DAD


class FeaturePropagator(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagator, self).__init__()
        self.num_iterations = num_iterations
        self.adaptive_diffusion = False

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor,edge_weight: OptTensor = None) -> Tensor:
        # out is initialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = None
        for _ in range(self.num_iterations):
            if self.adaptive_diffusion or adj is None:
                adj = self.get_propagation_matrix(out, edge_index, edge_weight, n_nodes)
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, edge_weight, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_weight = edge_weight if edge_weight else torch.ones(edge_index.shape[1]).to(edge_index.device)
        edge_weight = get_symmetrically_normalized_adjacency(edge_index, num_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight).to(edge_index.device)

        return adj
