import math
import numpy as np
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=1024,
        d_model=512,
        ):
        
        super().__init__()
        
        self.scaling_factor = scaling_factor
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            d_model=d_model,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
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
        # lsc = torch.norm(x - x_hat, p="fro") / torch.norm(x, p="fro")
        # lmag = torch.nn.functional.l1_loss(x, x_hat)
        return torch.nn.functional.mse_loss(x, x_hat)


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, d_model):
        super().__init__()
        
        self.enc_layers = nn.ModuleList()
        n_enc_layers = int(math.log(scaling_factor, 2))
        in_channels = 1
        for ith in range(n_enc_layers):
            self.enc_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels *= 2
            
        self.proj_out = nn.Linear(scaling_factor, d_model)
            
    def forward(self, x):
        # x: (b, l) -> (b, l, d)
        
        self.enc_out.clear()
        x = x.unsqueeze(1)
        
        for ith, layer in enumerate(self.enc_layers):
            x = nn.functional.relu(layer(x))
            self.enc_out = [x] + self.enc_out
            
        x = x.transpose(1, 2)
        x = self.proj_out(x)
        
        return x
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, d_model):
        super().__init__()
        
        self.proj_in = nn.Linear(d_model, scaling_factor)
        
        n_dec_layers = int(math.log(scaling_factor, 2))
        self.dec_layers = nn.ModuleList()
        in_channels = scaling_factor
        for ith in range(n_dec_layers):
            self.dec_layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = in_channels // 2

    def forward(self, x, enc_out):
        # x: (b, l, d) -> (b, l)
        
        x = self.proj_in(x)
        x = x.transpose(1, 2)
        
        for ith, layer in enumerate(self.dec_layers):
            x = x + enc_out[ith]
            x = layer(x)
        
        x = x.squeeze(1)
        
        return x
    
    
class SEEncoderLayer(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
          
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_out = nn.Conv2d(
            in_channels=inter_channels,
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
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        
        self.conv_in =  nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_out = nn.ConvTranspose2d(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        weight_scaling_init(self.conv_in)
        weight_scaling_init(self.conv_out)
    
    def forward(self, x):
        # x: (b, c, t, d)

        x = nn.functional.glu(self.conv_in(x), dim=1)
        x = self.conv_out(x)
        
        return x
        

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