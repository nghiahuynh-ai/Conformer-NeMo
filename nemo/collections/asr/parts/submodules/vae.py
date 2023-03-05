import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, MultiHeadAttention


class VAESpeechEnhance(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        n_encoder_layers=4,
        n_decoder_layers=4,
        feat_in=80,
        d_model=80,
        n_heads=8,
        self_attention_model='abs_pos',
        dropout=0.1,
        real_noise_filepath=None,
        real_noise_snr=[0, 5],
        white_noise_mean=0.0,   
        white_noise_std=[0.0, 0.05],
        ):
        
        super().__init__()
        
        self.proj_in = nn.Linear(feat_in, d_model)
        if self_attention_model == 'abs_pos':
            self.pos_enc = PositionalEncoding(d_model=d_model, dropout_rate=dropout)
            self.pos_enc.extend_pe(length=5000, device=next(self.parameters()).device)
        self.encoder = VAEEncoder(
                            latent_dim=latent_dim,
                            n_layers=n_encoder_layers,
                            d_model=d_model,
                            n_heads=n_heads,
                            self_attention_model=self_attention_model,
                            dropout=dropout,
                            )
        self.decoder = VAEDecoder(
                            latent_dim=latent_dim,
                            n_layers=n_decoder_layers,
                            d_model=d_model,
                            n_heads=n_heads,
                            self_attention_model=self_attention_model,
                            dropout=dropout,
                            )
        self.proj_out = nn.Linear(d_model, feat_in)
        
        self.add_noise_methods = []
        if real_noise_filepath is not None:
            self.add_noise_methods.append(self._add_real_noise)
            self.real_noise_corpus = np.load(real_noise_filepath, allow_pickle=True)
            self.real_noise_snr = real_noise_snr
        if white_noise_std[0] >= 0.0 and white_noise_std[1] >= white_noise_std[0]:
            self.add_noise_methods.append(self._add_white_noise)
            self.white_noise_mean = white_noise_mean
            self.white_noise_std = white_noise_std
        
        self.loss_fn = nn.MSELoss()
        self.loss_value = float('inf')
    
    def add_noise(self, signal):
        method = np.random.choice(self.add_noise_methods, size=1)
        return method(signal)
    
    def _add_real_noise(self, signal):
        signal_length = len(signal)
  
        # extract noise from noise list
        noise = np.random.choice(self.real_noise_corpus, size=1)
        start = np.random.randint(0, len(noise) - signal_length - 1)
        noise = torch.from_numpy(noise[start:start + signal_length]).to(signal.device)
        
        # calculate power of audio and noise
        snr = torch.randint(low=self.real_noise_snr[0], high=self.real_noise_snr[1], size=(1,))
        signal_energy = torch.mean(signal**2)
        noise_energy = torch.mean(noise**2)
        coef = torch.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
        signal_coef = torch.sqrt(1 / (1 + coef**2)).to(signal.device)
        noise_coef = torch.sqrt(coef**2 / (1 + coef**2)).to(signal.device)
        
        return signal_coef * signal + noise_coef * noise
    
    def _add_white_noise(self, signal):
        std = np.random.uniform(self.white_noise_std[0], self.white_noise_std[1])
        noise = np.random.normal(self.white_noise_mean, std, size=signal.shape)
        noise = torch.from_numpy(noise).type(torch.FloatTensor)
        return signal + noise.to(signal.device)
    
    def forward(self, x_noise, x_clean=None):
        x_noise = x_noise.transpose(-1, -2)
        if x_clean is not None:
            x_clean = x_clean.transpose(-1, -2)
        
        x = self.proj_in(x_noise)
        x = self.pos_enc(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.transpose(-1, -2)
        
        if self.training:
            self.loss_value = self.loss_fn(x_clean, x_hat) + self.encoder.kl
            
        return self.proj_out(x_hat)


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
        self.sigma = nn.Linear(d_model, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = float('inf')
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = mu + sigma * self.N.sample(mu.shape).to(x.device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
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
        

class VAEMHSALayer(nn.Module):
    def __init__(self, self_attention_model, d_model, n_heads, dropout):
        super().__init__()
        
        if self_attention_model == 'abs_pos':
            self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
        self.att_norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        x = self.att(x)
        x = self.att_norm(x)
        residual = residual + self.dropout(x)
        
        x = self.ff(residual)
        x = self.ff_norm(x)
        residual = residual + self.dropout(x)
        
        return nn.ReLU(residual)