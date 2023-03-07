import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class VAESpeechEnhance(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        downsize_factor=4,
        subsampling_factor=8,
        n_decoder_layers=8,                                                       
        hidden_shape=(0, 0),
        d_model=512,
        n_heads=8,
        ):
        
        super().__init__()
        
        self.conv_in = nn.ModuleList()
        in_channels = 1
        out_channels = d_model
        n_conv_layers = int(math.log(downsize_factor, 2))
        for ith, _ in enumerate(range(n_conv_layers)):
            if ith == n_conv_layers - 1:
                out_channels = 1
            self.conv_in.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels
        
        flatten_dim = int((hidden_shape[0] * hidden_shape[1]) / downsize_factor**2)
        self.flatten = nn.Flatten()
        self.mu = nn.Linear(flatten_dim, latent_dim)
        self.log_sigma = nn.Linear(flatten_dim, latent_dim)
        # self.decoder = VAEDecoder(
        #     latent_dim=latent_dim,
        #     flatten_dim=flatten_dim,
        #     hidden_shape=(int(hidden_shape[0]/downsize_factor), int(hidden_shape[1]/downsize_factor)),
        #     n_layers=n_decoder_layers,
        #     d_model=d_model,
        #     n_heads=n_heads,
        # )
        
        self.proj = nn.Linear(latent_dim, flatten_dim)
        self.unflatten = Unflatten(hidden_shape)
        
        self.upsampling = VAEUpsampling(downsize_factor * subsampling_factor, d_model)
        
        self.N = torch.distributions.Normal(0, 1)
        
        self.loss_fn = nn.MSELoss()
        self.kld = None
    
    def forward(self, x):

        x = x.transpose(2, 1)

        x = x.unsqueeze(1)
        for layer in self.conv_in:
            x = layer(x)
        x = x.squeeze(1)
            
        x = self.flatten(x)

        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        z = mu + sigma * self.N.sample(mu.shape).to(x.device)
        
        self.kld = -0.5 * torch.sum(1 - sigma**2 - mu**2 + log_sigma)

        # x_hat = self.decoder(z)
        x_hat = self.proj(z)
        x_hat = self.unflatten(x_hat)
        x_hat = self.upsampling(x_hat)
        
        x_hat = x_hat.transpose(2, 1)
        
        return x_hat
    
    def compute_loss(self, x_clean, x_hat):
        return self.loss_fn(x_clean, x_hat) + self.kld
    
  
class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        flatten_dim=None,
        hidden_shape=(0, 0),
        n_layers=8,
        d_model=512,
        n_heads=8,
        self_attention_model='abs_pos',
        dropout=0.1,
        ):
        super().__init__()
        
        self.proj_in = nn.Linear(latent_dim, flatten_dim)
        self.unflatten = Unflatten(hidden_shape)
        self.proj_att = nn.Linear(hidden_shape[1], d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                VAEMHSALayer(self_attention_model, d_model, n_heads, dropout)
            )
        self.proj_out = nn.Linear(d_model, flatten_dim)
        
    def forward(self, x):
        
        x_hat = self.proj_in(x)
        x_hat = self.unflatten(x_hat)
        x_hat = self.proj_att(x_hat)
        for layer in self.layers:
            x_hat = layer(x_hat)
        
        return self.proj_out(x_hat)
    
    
class VAEUpsampling(nn.Module):
    def __init__(self, upsampling_factor, conv_channel):
        super().__init__()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(upsampling_factor, 2))
        in_channels = 1
        out_channels = conv_channel
        for ith, _ in enumerate(range(n_layers)):
            if ith == n_layers - 1:
                out_channels = 1
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            in_channels = out_channels
            
    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)
        return x

class VAEMHSALayer(nn.Module):
    def __init__(self, self_attention_model, d_model, n_heads, dropout):
        super().__init__()
        
        if self_attention_model == 'abs_pos':
            self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
        self.att_norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.att(query=x, key=x, value=x, mask=None)
        x = self.att_norm(x)
        residual = residual + self.dropout(x)
        
        x = self.ff(residual)
        x = self.ff_norm(x)
        residual = residual + self.dropout(x)
        
        return self.activation(residual)

class Unflatten(nn.Module):
    def __init__(self, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape
        
    def forward(self, x):
        return x.reshape(x.size(0), self.unflatten_shape[0], self.unflatten_shape[1])