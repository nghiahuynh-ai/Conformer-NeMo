import math
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self, 
        downsampling_factor, 
        flatten_dim,
        latent_dim, 
        in_channels=1, 
        out_channels=64, 
        activation=nn.GELU()
        ):
        
        super().__init__()

        n_layers = int(math.log(downsampling_factor, 2))
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.layers.append(activation)
            in_channels = out_channels
        self.flatten = nn.Flatten()
        
        self.proj = nn.Linear(flatten_dim * out_channels, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.sigma = nn.Linear(latent_dim, latent_dim)
        
        self.distribution = torch.distributions.Normal(0, 1)
        
        self.kl = None
        self.old_shape = None
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        self.old_shape = x.shape
        x = self.flatten(x)
        x = self.proj(x)
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma * self.distribution.sample(mu.shape).to(mu.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        return z

class Decoder(nn.Module):
    def __init__(
        self, 
        upsampling_factor, 
        flatten_dim,
        latent_dim,
        in_channels=64, 
        out_channels=1, 
        activation=nn.GELU()
        ):
        
        super().__init__()
        
        self.proj = nn.Linear(latent_dim, flatten_dim * in_channels)
        
        n_layers = int(math.log(upsampling_factor, 2))
        self.layers = nn.ModuleList()
        
        for ith, _ in enumerate(range(n_layers)):
            _out_channels = in_channels if ith < n_layers - 1 else out_channels
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            self.layers.append(activation)
            
    def forward(self, x, old_shape):
        x = self.proj(x)
        x = x.reshape(old_shape)
        x = self.layers(x)
        
        return torch.squeeze(x, 1)


class SpeeechEnhance(nn.Module):
    def __init__(
        self,
            seq_len,
            freq_len,
            downsampling_factor=16,
            latent_dim=512,
        ):
        
        super().__init__()
        
        assert downsampling_factor % 2 == 0
        self.seq_len = seq_len
        
        flatten_dim = int(seq_len/downsampling_factor * freq_len/downsampling_factor)
        
        self.encoder = Encoder(
            downsampling_factor=downsampling_factor,
            flatten_dim=flatten_dim,
            latent_dim=latent_dim,
            )
        self.decoder = Decoder(
            upsampling_factor=downsampling_factor,
            flatten_dim=flatten_dim,
            latent_dim=latent_dim,
        )
        
        self.mse = nn.MSELoss()
        self.loss = None
        
    def forward(self, x):
        
        original_seq_len = x.size(-1)
        if original_seq_len < self.seq_len:
            x = nn.functional.pad(x, (self.seq_len - original_seq_len), value=0.0)
        
        z = self.encoder(x)
        x_hat = self.decoder(z, self.encoder.old_shape)
        self.loss = self.mse(x, x_hat) + self.encoder.kl
        
        x = x_hat[:,:,:original_seq_len]
        
        return x