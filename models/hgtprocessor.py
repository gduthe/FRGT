import torch
import torch.nn as nn
from torch_geometric.utils import unbatch
from .building_blocks import MPNNBlock, TransformerBlock

class HGTProcessor(nn.Module):
    ''' Hybrid Graph Transformer processor.
    kwargs:
        mp_type (str): type of message-passing layer.
        num_mp_steps (int): number of message-passing steps.
        num_gt_steps (int): number of graph transformer steps.
        node_latent_dim (int): dimension of node latent vectors.
        edge_latent_dim (int): dimension of edge latent vectors.
        dropout (float): dropout rate.
        attn_type (str): type of attention layer.
        num_heads (int): number of attention heads.
        dim_head (int): dimension of attention heads.
        mlp_dim (int): dimension of MLP layers.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        assert {'mp_type', 'num_mp_steps', 'num_gt_steps', 'latent_dim',  'dropout', 'attn_type', 'num_heads', 'dim_head'}.issubset(kwargs)

        # initialize the message-passing blocks
        self.mp_layers = nn.ModuleList()
        for i in range(kwargs['num_mp_steps']):
            mp_layer = MPNNBlock(node_latent_dim=kwargs['latent_dim'], edge_latent_dim=kwargs['latent_dim'], mp_type=kwargs['mp_type'])
            # self.mp_layers.append(GroupAddRev(mp_layer, num_groups=kwargs['num_groups']))
            self.mp_layers.append(mp_layer)

        # initialize the final graph transformer layer
        self.gt_layers = nn.ModuleList()
        for i in range(kwargs['num_gt_steps']):
            gt_layer = TransformerBlock(attn_type=kwargs['attn_type'], dim=kwargs['latent_dim'], heads=kwargs['num_heads'], 
                                    dim_head=kwargs['dim_head'], mlp_dim=kwargs['latent_dim'], qkv_bias=False, 
                                    drop=kwargs['dropout'], attn_drop=kwargs['dropout'], attn_layer_norm=True, mlp_layer_norm=True,)
            self.gt_layers.append(gt_layer)
        
        self.dropout = nn.Dropout(p=kwargs['dropout'])

    def forward(self, x, edge_index, edge_attr, batch):
        
        # update the node latent features through the message-passing
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
            x = self.dropout(x)
            
        # unsqueeze first dim for transformer layers
        x = x.unsqueeze(0)
        
        # run through the graph transformer layers
        for gt_layer in self.gt_layers:
            x = gt_layer(x, batch)

        # squeeze again the first dim
        x = x.squeeze(0)
        
        return x
    
class HGTProcessorIT(nn.Module):
    ''' Intertwined Hybrid Graph Transformer
    kwargs:
        mp_type (str): type of message-passing layer.
        num_mp_steps (int): number of message-passing steps.
        num_gt_steps (int): number of graph transformer steps.
        node_latent_dim (int): dimension of node latent vectors.
        edge_latent_dim (int): dimension of edge latent vectors.
        dropout (float): dropout rate.
        attn_type (str): type of attention layer.
        num_heads (int): number of attention heads.
        dim_head (int): dimension of attention heads.
        mlp_dim (int): dimension of MLP layers.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        assert {'mp_type', 'num_mp_steps_per_layer', 'num_layers', 'latent_dim', 'dropout', 'attn_type', 'num_heads', 'dim_head'}.issubset(kwargs)


        # initialize the message-passing blocks
        self.layers = nn.ModuleList()
        for i in range(kwargs['num_layers']):
            for j in range(kwargs['num_mp_steps_per_layer']):
                mp_layer = MPNNBlock(node_latent_dim=kwargs['latent_dim'], edge_latent_dim=kwargs['latent_dim'], mp_type=kwargs['mp_type'])
                self.layers.append(mp_layer)

            # initialize the final graph transformer layer
            gt_layer = TransformerBlock(attn_type=kwargs['attn_type'], dim=kwargs['latent_dim'], heads=kwargs['num_heads'], 
                                    dim_head=kwargs['dim_head'], mlp_dim=kwargs['latent_dim'], qkv_bias=False, 
                                    drop=kwargs['dropout'], attn_drop=kwargs['dropout'], attn_layer_norm=True, mlp_layer_norm=True,)
            self.layers.append(gt_layer)

        
        self.dropout = kwargs['dropout']

    def forward(self, x, edge_index, edge_attr, batch):
        # update the node latent features through the layers
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                x = x.unsqueeze(0) # format the 2d node features to 3d features for the graph transformer layer
                x = layer(x, batch)
                x = x.squeeze(0) # format the 3d node features back to 2d features
            else:
                x = layer(x, edge_index, edge_attr)
        return x

