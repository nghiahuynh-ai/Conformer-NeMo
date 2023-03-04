import torch
import torch.nn as nn


class VAEENcoder(nn.Module):
    def __init__(self):
        super().__init__(
            self,
            latent_dim=512,
            n_layers=4,
            feat_in=-1,
            d_model=-1,
            n_heads=8,
            self_attention_model='abs_pos',
            ff_expansion_factor=4,
            norm_type='layer_norm',
            dropout=0.1,
        )
    
    
class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class VAE(nn.Module):
    def __init__(self):
        super().__init__()