import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class VAESpeechEnhance(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        n_decoder_layers = 8,                                                       
        hidden_length=None,
        d_model=512,
        n_heads=8,
        ):
        
        super().__init__()
        
        self.hidden_shape = (hidden_length, d_model)
        self.flatten_dim = hidden_length * d_model
        self.latent_dim = latent_dim
        
        self.flatten = nn.Flatten()
        self.mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.log_sigma = nn.Linear(self.flatten_dim, self.latent_dim)
        self.unflatten = Unflatten(self.hidden_shape)
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            n_layers=n_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
        )
        
        self.loss_fn = nn.MSELoss()
        self.kld = None
        self.loss_value = None
    
    def forward(self, x_noise):
        x = self.proj_in(x_noise)
        x = self.pos_enc(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = self.proj_out(x_hat)
        x_hat = x_hat.transpose(-1, -2)
        
        return x_hat
    
    def compute_loss(self, x_clean, x_hat):
        return self.loss_fn(x_clean, x_hat) + self.encoder.kl


class VAEEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        n_layers=4,
        d_model=80,
        n_heads=8,
        self_attention_model='abs_pos',
        dropout=0.1,
        ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                VAEMHSALayer(self_attention_model, d_model, n_heads, dropout)
            )
        self.mu = nn.Linear(d_model, latent_dim)
        nn.init.xavier_uniform_(self.mu.weight, 0.04)
        # nn.init.zeros_(self.mu.weight)
        self.log_sigma = nn.Linear(d_model, latent_dim)
        nn.init.xavier_uniform_(self.log_sigma.weight, 0.04)
        # nn.init.zeros_(self.mu.weight)
        self.N = torch.distributions.Normal(0, 1)
        # self.activation = nn.ReLU()
        self.kl = None
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        z = mu + sigma * self.N.sample(mu.shape).to(x.device)
        self.kl = -0.5 * torch.sum(1 - sigma**2 - mu**2 + log_sigma)
        return z
  
class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        n_layers=4,
        d_model=80,
        n_heads=8,
        self_attention_model='abs_pos',
        dropout=0.1,
        ):
        super().__init__()
        
        self.proj = nn.Linear(latent_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight, 0.04)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                VAEMHSALayer(self_attention_model, d_model, n_heads, dropout)
            )
        
    def forward(self, x):
        x_hat = self.proj(x)
        for layer in self.layers:
            x_hat = layer(x_hat)
        return x_hat
    
    
class VAEUpsampling(nn.Module):
    def __init__(self, upsampling_factor):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(upsampling_factor):
            self.layers.append(
                nn.ConvTranspose2d()
            )
            
    def forward(self, x):
        
        

class VAEMHSALayer(nn.Module):
    def __init__(self, self_attention_model, d_model, n_heads, dropout):
        super().__init__()
        
        if self_attention_model == 'abs_pos':
            self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
            nn.init.xavier_uniform_(self.att.linear_q.weight, 0.04)
            nn.init.xavier_uniform_(self.att.linear_k.weight, 0.04)
            nn.init.xavier_uniform_(self.att.linear_v.weight, 0.04)
            nn.init.xavier_uniform_(self.att.linear_out.weight, 0.04)
        self.att_norm = nn.LayerNorm(d_model)
        # nn.init.xavier_uniform_(self.att_norm.weight, 0.04)
        self.ff = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.ff.weight, 0.04)
        self.ff_norm = nn.LayerNorm(d_model)
        # nn.init.xavier_uniform_(self.ff_norm.weight, 0.04)
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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model) 

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor):
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :]) * math.sqrt(self.d_model)
    

class Unflatten(nn.Module):
    def __init__(self, unflatten_shape):
        super().__init__()
        self.unflatten_shape = unflatten_shape
        
    def forward(self, x):
        return x.reshape(x.size(0), self.unflatten_shape[0], self.unflatten_shape[1])