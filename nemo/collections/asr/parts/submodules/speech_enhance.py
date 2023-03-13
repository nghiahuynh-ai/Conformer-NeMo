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
        expand_factor=8,
        ):
        
        super().__init__()
        
        self.n_features = n_features
        self.scaling_factor = scaling_factor
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            dim_in=n_features,
            dim_out=asr_d_model,
            dim_expand=expand_factor * n_features,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            dim_in=asr_d_model,
            dim_out=n_features,
            dim_narrow=int(expand_factor / scaling_factor * n_features)
        )
        
    def forward_encoder(self, x):
        return self.encoder(x), torch.tensor([int(x / self.scaling_factor)], dtype=torch.int, device=x.device)
    
    def forward_decoder(self, x):
        return self.decoder(x, self.encoder.layers_out)
    
    def compute_loss(self, x, x_hat):
        # x, x_hat: (b, t, d)
        lsc = torch.norm(x - x_hat, p="fro") / torch.norm(x, p="fro")
        lmag = torch.nn.functional.l1_loss(x, x_hat)
        return lsc + lmag


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out, dim_expand):
        super().__init__()
        
        self.proj_in = nn.Linear(dim_in, dim_expand)
        self.act_in = nn.ReLU()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            self.layers.append(
                SEConvModule(
                    dim_in=int(dim_expand / 2**ith),
                    dim_out=int(dim_expand / 2**(ith + 1)),
                    conv_channels=conv_channels,
                )
            )
            
        self.layers_out = []
        
        self.proj_out = nn.Linear(int(dim_expand / scaling_factor), dim_out)
        self.act_out = nn.ReLU()
            
    def forward(self, x):
        # x: (b, t, d)
        
        self.layers_out.clear()
        
        x = self.proj_in(x)
        x = self.act_in(x)
        
        for layer in self.layers:
            x = layer(x)
            self.layers_out = [x] + self.layers_out
        
        x = self.proj_out(x)
        
        return self.act_out(x)
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out, dim_narrow):
        super().__init__()
        
        self.proj_in = nn.Linear(dim_in, dim_narrow)
        self.act_in = nn.ReLU()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            self.layers.append(
                SEConvTransposedModule(
                dim_in=int(dim_narrow * 2**ith),
                dim_out=int(dim_narrow * 2**(ith + 1)),
                conv_channels=conv_channels,
                )
            )
            
        self.proj_out = nn.Linear(int(dim_narrow * scaling_factor), dim_out)
        self.act_out = nn.ReLU()
            
    def forward(self, x, enc_out):
        # x: (b, t, d)

        x = self.proj_in(x)
        x = self.act_in(x)
        
        for ith, layer in enumerate(self.layers):
            x = x + enc_out[ith]
            x = layer(x)
        
        x = self.proj_out(x)
        
        return self.act_out(x)
    
    
class SEConvModule(nn.Module):
    def __init__(self, dim_in, dim_out, conv_channels):
        super().__init__()
        
        self.conv_in = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        self.norm_in = nn.BatchNorm2d(conv_channels)
        self.conv_out = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            )
        self.norm_out = nn.BatchNorm2d(conv_channels)
        self.proj_out = nn.Linear(conv_channels * int(dim_in / 2), dim_out)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: (b, t, d) -> (b, 1, t, d)

        x = x.unsqueeze(1)
        x = self.conv_in(x)
        x = self.norm_in(x)
        x = self.conv_out(x)
        x = self.norm_out(x)
        b, c, t, d = x.shape
        x = x.reshape(b, t, c * d)
        
        x = self.proj_out(x)
        return self.activation(x)
    
    
class SEConvTransposedModule(nn.Module):
    def __init__(self, dim_in, dim_out, conv_channels):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            )
        self.norm_in = nn.BatchNorm2d(conv_channels)
        self.conv_out = nn.ConvTranspose2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            )
        self.norm_out = nn.BatchNorm2d(conv_channels)
        self.proj_out = nn.Linear(conv_channels * (2 * dim_in), dim_out)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: (b, t, d) -> (b, 1, t, d)

        x = x.unsqueeze(1)
        x = self.conv_in(x)
        x = self.norm_in(x)
        x = self.conv_out(x)
        x = self.norm_out(x)
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

# class PositionalEncoding2D(nn.Module):
#     def __init__(self, channels):
#         """
#         :param channels: The last dimension of the tensor you want to apply pos emb to.
#         """
#         super(PositionalEncoding2D, self).__init__()
#         self.org_channels = channels
#         channels = int(np.ceil(channels / 4) * 2)
#         self.channels = channels
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
#         self.register_buffer("inv_freq", inv_freq)
#         self.cached_penc = None

#     def forward(self, tensor):
#         """
#         :param tensor: A 4d tensor of size (batch_size, x, y, ch)
#         :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
#         """
#         if len(tensor.shape) != 4:
#             raise RuntimeError("The input tensor has to be 4d!")

#         if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
#             return self.cached_penc

#         self.cached_penc = None
#         batch_size, x, y, orig_ch = tensor.shape
#         pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
#         pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
#         sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
#         emb_x = get_emb(sin_inp_x).unsqueeze(1)
#         emb_y = get_emb(sin_inp_y)
#         emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
#             tensor.type()
#         )
#         emb[:, :, : self.channels] = emb_x
#         emb[:, :, self.channels : 2 * self.channels] = emb_y

#         self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
#         return self.cached_penc


# def get_emb(sin_inp):
#     """
#     Gets a base embedding for one dimension with sin and cos intertwined
#     """
#     emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
#     return torch.flatten(emb, -2, -1)