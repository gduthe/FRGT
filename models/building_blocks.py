import torch
from torch import nn
from einops import rearrange
from torch_geometric.nn import GATConv, GENConv, GINEConv
from torch_geometric.nn import MLP


class MPNNBlock(nn.Module):
    def __init__(self, node_latent_dim, edge_latent_dim, mp_type='GAT'):
        super().__init__()
        assert (mp_type in ['GAT', 'GEN', 'GINE'])

        if mp_type == 'GAT':
            self.conv = GATConv(in_channels=node_latent_dim, out_channels=node_latent_dim, edge_dim=edge_latent_dim, heads=2, add_self_loops=True, concat=False)
        elif mp_type == 'GEN':
            self.conv = GENConv(in_channels=node_latent_dim, out_channels=node_latent_dim, norm='layer', aggr='softmax', num_layers=2)
        elif mp_type == 'GINE':
            gin_mlp =   MLP(in_channels=node_latent_dim, hidden_channels=node_latent_dim*2, out_channels=node_latent_dim,
                            num_layers=2, act='relu', norm='layer')
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

    def forward(self, x, edge_index, edge_attr):
        x = x + self.norm(self.conv(x, edge_index, edge_attr))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, attn_type, dim, heads, dim_head, mlp_dim, qkv_bias=False, drop=0., attn_drop=0., attn_layer_norm=True, mlp_layer_norm=True):
        super().__init__()
        
        assert attn_type in {'classic', 'galerkin'}, 'Attention type must be either classic, galerkin or fourier'
            
        if attn_type == 'classic':
            self.attn = ClassicAttention(dim, heads, dim_head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer_norm=attn_layer_norm)
        elif attn_type == 'galerkin':
            self.attn = GalerkinAttention(dim, heads, dim_head, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer_norms=attn_layer_norm)

        self.mlp_layer_norm  = nn.LayerNorm(dim, eps=1e-6) if mlp_layer_norm else nn.Identity()
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=drop)
        
    def forward(self, x, batch, return_internal=False):        
        if return_internal:
            z, attn_, q, k, v, z_pre_proj = self.attn(x, return_qkvz=True)
        else:
            z = self.attn(x, batch)
                
        z_add_x = z  + x
            
        z_ff = self.mlp(self.mlp_layer_norm(z_add_x))
        x = z_ff + z_add_x
        
        if return_internal:
            return x, attn_, q, k, v, z_pre_proj, z, z_ff, z_add_x
        else:
            return x
        

class ClassicAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, qkv_bias=False, attn_drop=0., proj_drop=0., layer_norm=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.norm = nn.LayerNorm(dim, eps=1e-6) if layer_norm else nn.Identity()

        self.heads = heads
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
                
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if project_out else nn.Identity()

    def forward(self, x, batch, return_qkvz=False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Create mask from batch indices
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop if self.training else 0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_qkvz:
            return self.to_out(out), q, k, v, out
        else:
            return self.to_out(out)


class GalerkinAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, attn_drop=0., proj_drop=0., layer_norms=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        if layer_norms:
            self.norm1 = nn.LayerNorm(dim_head, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim_head, eps=1e-6)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if project_out else nn.Identity()

    def forward(self, x, batch, return_qkvz=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        k = self.norm1(k)
        v = self.norm2(v)
        
        # process each graph separately
        start_idx = 0
        out = torch.zeros_like(q)

        # compute the size of each graph based on the batch index
        graph_sizes = torch.bincount(batch).tolist()
        
        for size in graph_sizes:
            end_idx = start_idx + size
            
            # Compute attention only within each graph
            graph_k = k[..., start_idx:end_idx, :]
            graph_v = v[..., start_idx:end_idx, :]
            graph_q = q[..., start_idx:end_idx, :] / size
            
            ktv = torch.matmul(graph_k.permute(0,1,3,2), graph_v)
            out[..., start_idx:end_idx, :] = torch.matmul(graph_q, ktv)
            start_idx = end_idx
            
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_qkvz:
            return self.to_out(out), q, k, v, out
        else:
            return self.to_out(out)
        
    def forward_(self, x, batch, return_qkvz=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        k = self.norm1(k)
        v = self.norm2(v)
        
        
        # scale q by dividing by n
        q = q/n
        
        ktv = torch.matmul(k.permute(0,1,3,2), v)
        out = torch.matmul(q, ktv)
            
        out = rearrange(out, 'b h n d -> b n (h d)')

        if return_qkvz:
            return self.to_out(out), ktv, q, k, v, out
        else:
            return self.to_out(out)