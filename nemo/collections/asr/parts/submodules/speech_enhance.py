import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=8,
        conv_channels=80,
        n_layers=16,
        d_model=512,
        n_heads=8,
        ):
        
        super().__init__()
        
        self.scaling_factor = scaling_factor
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
        )
        
        self.pos_enc = PositionalEncoding1D(conv_channels)
        
        self.bottleneck = SEBottleNeck(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            dim_in=conv_channels,
            dim_out=conv_channels,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
        )
    
    def forward(self, x):
        x_hat = self.encoder(x)
        x_hat = self.pos_enc(x_hat)
        x_hat = self.bottleneck(x_hat)
        x_hat = self.decoder(x_hat)
        return x_hat
    
    def compute_loss(self, x, x_hat):
        # lsc = torch.norm(x - x_hat, p="fro") / torch.norm(x, p="fro")
        # lmag = torch.nn.functional.l1_loss(x, x_hat)
        return torch.nn.functional.mse_loss(x, x_hat)


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels):
        super().__init__()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            if ith == 0:
                in_channels = 1
                out_channels = conv_channels
            else:
                in_channels = conv_channels
                out_channels = conv_channels
            self.layers.append(
                SEEncoderLayer(in_channels=in_channels, out_channels=out_channels)
            )
            
    def forward(self, x):
        # x: (b, l) -> (b, l, c)
        
        x = x.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        
        return x
    
    
class SEBottleNeck(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dim_in, dim_out):
        super().__init__()
        
        self.proj_in = nn.Linear(dim_in, d_model)
        self.layers = nn.ModuleList()
        for ith in range(n_layers):
            self.layers.append(
                SETransformerLayer(d_model=d_model, n_heads=n_heads)
            )
        self.proj_out = nn.Linear(d_model, dim_out)
        
    def forward(self, x):
        #x: (b, l, d)
        
        x = self.proj_in(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.proj_out(x)
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels):
        super().__init__()

        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            if ith == n_layers - 1:
                out_channels = 1
            else:
                out_channels = conv_channels
            self.layers.append(
                SEDecoderLayer(in_channels=conv_channels, out_channels=out_channels)
            )
        self.proj_out = nn.Linear()
            
    def forward(self, x):
        # x: (b, l, c) -> (b, l)
        
        x = x.transpose(1, 2)
        
        for ith, layer in enumerate(self.layers):
            x = layer(x)

        x = x.squeeze(1)
        
        return x
    
    
class SEEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            )
        
        self.conv_out = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            )
    
    def forward(self, x):
        # x: (b, c, l)
        
        x = nn.functional.relu(self.conv_in(x))
        x = nn.functional.glu(self.conv_out(x), dim=1)

        return x
    
    
class SEDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.conv_out = nn.ConvTranspose1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
    
    def forward(self, x):
        # x: (b, c, l)

        x = nn.functional.glu(self.conv_in(x), dim=1)
        x = self.conv_out(x)
        
        return x
        

class SETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.att_norm = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.att_norm(residual)
        x = self.att(query=x, key=x, value=x, mask=None)
        residual = residual + self.dropout(x)
        
        x = self.ff_norm(residual)
        x = self.ff(x)
        residual = residual + self.dropout(x)
        
        x = self.norm_out(x)
        return self.activation(residual)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)