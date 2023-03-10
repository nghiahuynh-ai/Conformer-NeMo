import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=8,
        n_features=80,
        asr_d_model=512,
        conv_channels=64,
        d_model=512,
        n_heads=8,
        ):
        
        super().__init__()
        
        self.n_features = n_features
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            dim_in=n_features,
            dim_out=asr_d_model,
            d_model=d_model,
            n_heads=n_heads,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            dim_in=asr_d_model,
            dim_out=n_features,
            d_model=int(d_model / scaling_factor),
            n_heads=n_heads,
        )
        
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder(self, x):
        return self.decoder(x, self.encoder.layers_out)
    
    def compute_loss(self, x, x_hat):
        # x, x_hat: (b, t, d)
        lsc = torch.norm(x - x_hat, p="fro") / torch.norm(x, p="fro")
        lmag = torch.nn.functional.l1_loss(x, x_hat)
        
        return lsc + lmag

class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out, d_model, n_heads):
        super().__init__()
        
        self.norm_in = nn.LayerNorm(dim_in)
        self.proj_in = nn.Linear(dim_in, d_model)
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            self.layers.append(
                SEConvModule(
                    dim_in=int(d_model / 2**ith),
                    dim_out=int(d_model / 2**(ith + 1)),
                    conv_channels=conv_channels,
                )
            )
            self.layers.append(
                SETransModule(
                    d_model=int(d_model / 2**(ith + 1)),
                    n_heads=n_heads,
                )
            )
            
        self.layers_out = []
        
        self.proj_out = nn.Linear(int(d_model / scaling_factor), dim_out)
        self.act_out = nn.ReLU()
            
    def forward(self, x):
        # x: (b, t, d)
        
        x = self.norm_in(x)
        x = self.proj_in(x)
        
        for ith, layer in enumerate(self.layers):
            x = layer(x)
            if ith % 2 == 1:
                self.layers_out = [x] + self.layers_out
        
        x = self.proj_out(x)
        return self.act_out(x)
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out, d_model, n_heads):
        super().__init__()
        
        self.norm_in = nn.LayerNorm(dim_in)
        self.proj_in = nn.Linear(dim_in, d_model)
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            self.layers.append(
                SETransModule(
                    d_model=int(d_model * 2**ith),
                    n_heads=n_heads,
                )
            )
            self.layers.append(
                SEConvTransposedModule(
                dim_in=int(d_model * 2**ith),
                dim_out=int(d_model * 2**(ith + 1)),
                conv_channels=conv_channels,
                )
            )
            
        self.proj_out = nn.Linear(int(d_model * scaling_factor), dim_out)
        self.norm_out = nn.LayerNorm(dim_out)
        self.act_out = nn.ReLU()
            
    def forward(self, x, enc_out):
        # x: (b, t, d)
        
        x = self.norm_in(x)
        x = self.proj_in(x)
        
        for ith, layer in enumerate(self.layers):
            if ith % 2 == 0:
                x = enc_out[int(ith / 2)] + layer(x)
            else:
                x = layer(x)
        
        x = self.proj_out(x)
        x = self.norm_out(x)
        
        return self.act_out(x)
    
    
class SEConvModule(nn.Module):
    def __init__(self, dim_in, dim_out, conv_channels):
        super().__init__()
        
        self.norm_in = nn.LayerNorm(dim_in)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.proj_out = nn.Linear(conv_channels * int(dim_in / 2), dim_out)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: (b, t, d) -> (b, 1, t, d)
        
        x = self.norm_in(x)
        
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, d = x.shape
        x = x.reshape(b, t, c * d)
        
        x = self.proj_out(x)
        return self.activation(x)
    
    
class SEConvTransposedModule(nn.Module):
    def __init__(self, dim_in, dim_out, conv_channels):
        super().__init__()
        
        self.norm_in = nn.LayerNorm(dim_in)
        self.conv = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            )
        self.proj_out = nn.Linear(conv_channels * (2 * dim_in), dim_out)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: (b, t, d) -> (b, 1, t, d)
        
        x = self.norm_in(x)
        
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, d = x.shape
        x = x.reshape(b, t, c * d)
        
        x = self.proj_out(x)
        return self.activation(x)
        

class SETransModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.att_norm = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.att_norm(residual)
        x = self.att(query=x, key=x, value=x, mask=None)
        residual = residual + self.dropout(x)
        
        x = self.ff_norm(residual)
        x = self.ff(x)
        residual = residual + self.dropout(x)
        
        return self.activation(residual)