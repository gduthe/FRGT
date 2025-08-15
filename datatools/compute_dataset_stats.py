import numpy as np
import torch

def compute_dataset_stats(dataloader, device):
    ''' A method to efficiently compute the dataset stats for large graphs making use of iterative mean and std computation
        with a dataloader.
    params:
        dataloader: a dataloader for the dataset
        device: the device to use for the computations
    '''
    
    with torch.no_grad():
        # iterative variables for the input globals
        glob_in_max = []
        glob_in_min = []
        glob_in_sum = torch.zeros(dataloader.dataset.num_glob_features, requires_grad=False)
        glob_in_sum_squared = torch.zeros(dataloader.dataset.num_glob_features, requires_grad=False)
        
        # iterative variables for the output globals
        glob_out_max = []
        glob_out_min = []
        glob_out_sum = torch.zeros(dataloader.dataset.num_glob_output_features, requires_grad=False)
        glob_out_sum_squared = torch.zeros(dataloader.dataset.num_glob_output_features, requires_grad=False)

        # iterative variables for the edge attributes
        ea_min = []
        ea_max = []
        ea_sum = torch.zeros(dataloader.dataset.num_edge_features, requires_grad=False)
        ea_sum_squared = torch.zeros(dataloader.dataset.num_edge_features, requires_grad=False)

        # iterative variables for the y target values
        y_min = []
        y_max = []
        y_sum = torch.zeros(dataloader.dataset.num_node_output_features, requires_grad=False)
        y_sum_squared = torch.zeros(dataloader.dataset.num_node_output_features, requires_grad=False)

        # iterative variables for the x input values
        x_min = []
        x_max = []
        x_sum = torch.zeros(dataloader.dataset.num_node_features, requires_grad=False)
        x_sum_squared = torch.zeros(dataloader.dataset.num_node_features, requires_grad=False)

        n_nodes = 0
        n_edges = 0
        for data in dataloader:
            # compute batch input globals stats
            glob_in_min.append(data.globals.min(dim=0).values.tolist())
            glob_in_max.append(data.globals.max(dim=0).values.tolist())
            glob_in_sum += data.globals.sum(dim=0)
            glob_in_sum_squared += (data.globals ** 2).sum(dim=0)
            
            # compute batch output globals stats
            glob_out_min.append(data.globals_y.min(dim=0).values.tolist())
            glob_out_max.append(data.globals_y.max(dim=0).values.tolist())
            glob_out_sum += data.globals_y.sum(dim=0)
            glob_out_sum_squared += (data.globals_y ** 2).sum(dim=0)

            # compute batch edge_stats
            ea_min.append(data.edge_attr.min(dim=0).values.tolist())
            ea_max.append(data.edge_attr.max(dim=0).values.tolist())
            ea_sum += data.edge_attr.sum(dim=0)
            ea_sum_squared += (data.edge_attr ** 2).sum(dim=0)
            n_edges += data.edge_attr.shape[0]

            # compute batch y stats
            y_min.append(data.y.min(dim=0).values.tolist())
            y_max.append(data.y.max(dim=0).values.tolist())
            y_sum += data.y.sum(dim=0)
            y_sum_squared += (data.y**2).sum(dim=0)
            n_nodes += data.y.shape[0]

            # compute batch x ignoring the masked input values
            x_min.append(np.nanmin(data.x.numpy(), axis=0).tolist())
            x_max.append(np.nanmax(data.x.numpy(), axis=0).tolist())
            x_sum += np.nansum(data.x.numpy(), axis=0)
            x_sum_squared += np.nansum((data.x ** 2), axis=0)

        # save final global stats
        n_graphs = len(dataloader.dataset)
        glob_in_stats = {'max': torch.tensor(np.max(glob_in_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'min': torch.tensor(np.min(glob_in_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'mean': (glob_in_sum/n_graphs).to(device),
                      'std': torch.sqrt(glob_in_sum_squared/n_graphs - (glob_in_sum/n_graphs)**2).to(device)}
        
        glob_out_stats = {'max': torch.tensor(np.max(glob_out_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'min': torch.tensor(np.min(glob_out_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'mean': (glob_out_sum/n_graphs).to(device),
                      'std': torch.sqrt(glob_out_sum_squared/n_graphs - (glob_out_sum/n_graphs)**2).to(device)}

        # save final edge attributes stats
        edge_stats = {'max': torch.tensor(np.max(ea_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'min': torch.tensor(np.min(ea_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                      'mean': (ea_sum/n_edges).to(device),
                      'std': torch.sqrt(ea_sum_squared/n_edges - (ea_sum/n_edges)** 2).to(device)}

        # save final y stats
        y_stats = {'max': torch.tensor(np.max(y_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                   'min': torch.tensor(np.min(y_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                   'mean': (y_sum/n_nodes).to(device),
                   'std': torch.sqrt(y_sum_squared/n_nodes - (y_sum/n_nodes)**2).to(device)}

        # save final x stats
        x_stats = {'max': torch.tensor(np.max(x_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                    'min': torch.tensor(np.min(x_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                    'mean': (x_sum/n_nodes).to(device),
                    'std': torch.sqrt(x_sum_squared/n_nodes - (x_sum/n_nodes)** 2).to(device)}
        
        return {'x': x_stats, 'y': y_stats, 'edge_attrs': edge_stats, 'globals_in': glob_in_stats, 'globals_out': glob_out_stats}