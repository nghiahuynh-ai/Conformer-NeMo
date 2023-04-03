import math
import torch
import torch.nn as nn
from nemo.collections.asr.parts.submodules.multi_head_attention import MultiHeadAttention


class SpeechEnhance(nn.Module):
    def __init__(
        self,
        scaling_factor=4,
        n_features=80,
        asr_d_model=512,
        conv_channels=512,
        ):
        
        super().__init__()
        
        self.n_features = n_features
        self.scaling_factor = scaling_factor
        
        self.encoder = SEEncoder(
            scaling_factor=scaling_factor,
            conv_channels=asr_d_model,
            dim_in=n_features,
            dim_out=asr_d_model,
        )
        
        self.decoder = SEDecoder(
            scaling_factor=scaling_factor,
            conv_channels=conv_channels,
            dim_in=asr_d_model,
            dim_out=n_features,
        )
        
        # for layer in self.modules():
        #     if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        #         torch.nn.init.xavier_uniform_(layer.weight, gain=0.05)
        #         layer.bias.data.fill_(0.08)
        
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

    def forward_loss(self, x, x_hat, x_len):
        for i in range(len(x)):
            x_hat[i, :, x_len[i]:] = 0.0
        return torch.nn.functional.mse_loss(x, x_hat)


class SEEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out):
        super().__init__()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        in_channels = 1
        for ith in range(n_layers):
            self.layers.append(
                SEEncoderLayer(in_channels=in_channels, out_channels=conv_channels)
            )
            in_channels = conv_channels
        
        self.out_layers = nn.ModuleList()
        n_out_layers = int(math.log(dim_in // (scaling_factor * 5), 2))
        in_channels = dim_in // scaling_factor * conv_channels
        out_channels = in_channels // 2
        for ith in range(n_out_layers):
            self.out_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels
            out_channels = in_channels // 2
        
        self.enc_out = []
        self.proj_out = nn.Linear(5 * conv_channels, dim_out)
            
    def forward(self, x):
        # x: (b, t, d)
        print(x.shape)
        self.enc_out.clear()
        x = x.unsqueeze(1)
        print(x.shape)
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
            self.enc_out = [x] + self.enc_out
        print(x.shape)
        b, c, t, d = x.shape
        x = x.transpose(1, 2).reshape(b, -1, t)
        print(x.shape)
        for layer in self.out_layers:
            x = nn.functional.relu(layer(x))
            self.enc_out = [x] + self.enc_out
        print(x.shape)
        x = x.transpose(1, 2)
        print(x.shape)
        x = self.proj_out(x)
        print(x.shape)
        return x
        
    
class SEDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, dim_in, dim_out):
        super().__init__()
        
        self.conv_channels = conv_channels
        self.proj_in = nn.Linear(dim_in, conv_channels * 5)
        
        self.in_layers = nn.ModuleList()
        n_in_layers = int(math.log(dim_out // (scaling_factor * 5), 2))
        in_channels = conv_channels * 5
        out_channels = in_channels * 2
        for ith in range(n_in_layers):
            self.in_layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = out_channels
            out_channels = in_channels * 2
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            out_channels = 1 if ith == n_layers - 1 else conv_channels
            self.layers.append(
                SEDecoderLayer(in_channels=conv_channels, out_channels=out_channels)
            )
            
    def forward(self, x, enc_out):
        # x: (b, t, d)

        x = self.proj_in(x)
        x = x.transpose(1, 2)
        
        for layer in self.in_layers:
            x = enc_out.pop(0) + nn.functional.relu(layer(x))
        
        b, d, t = x.shape
        x = x.reshape(b, self.conv_channels, t, d // self.conv_channels)
        
        for ith, layer in enumerate(self.layers):
            x = x + enc_out.pop(0)
            x = layer(x)

        x = x.squeeze(1)
        x = x.transpose(1, 2)
        
        return x
    
    
class SEEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x):
        # x: (b, t, d)
        
        x = self.conv1(x)
        x = nn.functional.relu(self.batchnorm1(x))
        x = x + self.conv2(x)
        x = nn.functional.relu(self.batchnorm2(x))
        x = x + self.conv3(x)
        x = nn.functional.relu(self.batchnorm3(x))

        return x
    
    
class SEDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(num_features=in_channels)
        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
    
    def forward(self, x):
        # x: (b, c, t, d)

        x = x + self.conv1(x)
        x = nn.functional.relu(self.batchnorm1(x))
        x = x + self.conv2(x)
        x = nn.functional.relu(self.batchnorm2(x))
        x = self.conv3(x)
        
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