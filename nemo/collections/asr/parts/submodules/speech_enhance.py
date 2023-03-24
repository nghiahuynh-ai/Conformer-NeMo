import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=256,
        n_layers=4,
        d_model=512,
        n_heads=8,
        ):
        
        super().__init__()
        
        self.scaling_factor = scaling_factor
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            dim_out=d_model,
        )
        
        self.pos_enc = PositionalEncoding1D(d_model)
        self.bottleneck = SEBottleNeck(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
        )
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            dim_in=d_model,
        )
    
    def forward(self, x):
        x_hat = self.encoder(x)
        x_hat = self.pos_enc(x_hat)
        x_hat = self.bottleneck(x_hat)
        x_hat = self.decoder(x_hat, self.encoder.layers_out)
        return x_hat
    
    def compute_loss(self, x, x_hat):
        # lsc = torch.norm(x - x_hat, p="fro") / torch.norm(x, p="fro")
        # lmag = torch.nn.functional.l1_loss(x, x_hat)
        return torch.nn.functional.mse_loss(x, x_hat)


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, dim_out):
        super().__init__()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        in_channels = 1
        out_channels = 2
        for ith in range(n_layers):
            self.layers.append(
                SEEncoderLayer(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels
            out_channels *= 2
            
        self.conv_out = nn.Conv1d(in_channels=in_channels, out_channels=dim_out, kernel_size=1)
        self.layers_out = []
        
    def forward(self, x):
        # in: (b, l)
        # out: (b, l, c)
        
        std = x.std(dim=2, keepdim=True) + 1e-3
        x /= std
        
        self.layers_out.clear()
        
        x = x.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
            self.layers_out = [x] + self.layers_out

        x = self.conv_out(x)
        x = x.transpose(1, 2)
        
        return x
    
    
class SEBottleNeck(nn.Module):
    def __init__(self, n_layers, d_model, n_heads):
        super().__init__()

        self.layers = nn.ModuleList()
        for ith in range(n_layers):
            self.layers.append(
                SETransformerLayer(d_model=d_model, n_heads=n_heads)
            )
        
    def forward(self, x):
        # in: (b, l, c)
        # out: (b, l, c)
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, dim_in):
        super().__init__()

        self.conv_in = nn.Conv1d(dim_in, scaling_factor, kernel_size=1)

        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        in_channels = scaling_factor
        out_channels = int(scaling_factor / 2)
        for ith in range(n_layers):
            self.layers.append(
                SEDecoderLayer(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels
            out_channels = int(out_channels / 2)
            
    def forward(self, x, enc_out):
        # in: (b, l, c)
        # out: (b, l)
        
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        
        for ith, layer in enumerate(self.layers):
            x = x + enc_out[ith]
            x = layer(x)

        x = x.squeeze(1)
        
        return x
    
    
class SEEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            )
        self.conv_out = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            )
        weight_scaling_init(self.conv_in)
        weight_scaling_init(self.conv_out)
    
    def forward(self, x):
        # x: (b, t, d)
        
        x = nn.functional.relu(self.conv_in(x))
        x = nn.functional.glu(self.conv_out(x), dim=1)

        return x
    
    
class SEDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_out = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        weight_scaling_init(self.conv_in)
        weight_scaling_init(self.conv_out)
    
    def forward(self, x):
        # x: (b, t, d)

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
    

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
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


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)