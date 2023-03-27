import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=512,
        conv_channels=512,
        d_model=512,
        ):
        
        super().__init__()
        
        self.scaling_factor = scaling_factor
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            d_model=d_model,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            d_model=d_model,
        )
        
    def forward_encoder(self, x, length):
        length = calc_length(
            lengths=length,
            padding=1,
            kernel_size=4,
            ceil_mode=False,
            stride=2,
            repeat_num=int(math.log(self.scaling_factor, 2)),
        )
        return self.encoder(x), length
    
    def forward_decoder(self, x):
        return self.decoder(x, self.encoder.enc_out)
    
    def compute_loss(self, x, x_hat):
        return torch.nn.functional.mse_loss(x, x_hat)


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, d_model):
        super().__init__()
        
        self.enc_layers = nn.ModuleList()
        n_enc_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_enc_layers):
            in_channels = 1 if ith == 0 else in_channels * 2
            self.enc_layers.append(
                SEEncoderLayer(in_channels=in_channels, out_channels=in_channels * 2)
            )
        self.enc_out = []
            
        self.proj_out = nn.Linear(in_channels * 2, d_model)
            
    def forward(self, x):
        # x: (b, l) -> (b, l, d)
        
        # normalize
        std = x.std(dim=1, keepdim=True) + 1e-3
        x /= std
        
        self.enc_out.clear()
        x = x.unsqueeze(1)
        
        for ith, layer in enumerate(self.enc_layers):
            x = nn.functional.relu(layer(x))
            self.enc_out = [x] + self.enc_out
            
        x = x.transpose(1, 2)
        x = self.proj_out(x)
        
        return x
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, d_model):
        super().__init__()
        
        self.proj_in = nn.Linear(d_model, conv_channels)
        
        n_dec_layers = int(math.log(scaling_factor, 2))
        self.dec_layers = nn.ModuleList()
        for ith in range(n_dec_layers):
            in_channels = scaling_factor if ith == 0 else in_channels // 2
            self.dec_layers.append(
                SEDecoderLayer(in_channels=conv_channels, out_channels=in_channels // 2)
            )

    def forward(self, x, enc_out):
        # x: (b, d, l) -> (b, l)

        x = x.transpose(1, 2)
        x = self.proj_in(x)
        x = x.transpose(1, 2)
        
        for ith, layer in enumerate(self.dec_layers):
            x = x + enc_out[ith]
            x = layer(x)
        
        x = x.squeeze(1)
        
        return x
    
    
class SEEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
          
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        
        weight_scaling_init(self.conv)
    
    def forward(self, x):
        # x: (b, t, d)
        
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        
        return x
    
    
class SEDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv =  nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        weight_scaling_init(self.conv)
    
    def forward(self, x):
        return self.conv(x)
        

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