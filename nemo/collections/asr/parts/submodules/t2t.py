import torch
import torch.nn as nn


class Text2Text(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        pred_dim=320,
        ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.pred_dim = pred_dim
        
        self.t2t_model = nn.Transformer(
            d_model=d_model, 
            nhead=n_heads, 
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_model,
            batch_first=True,
            )
        
        self.t2t_out = nn.Linear(d_model, pred_dim)
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input, target, grad=True):
        
        batch_size = target.shape[0]
        tgt_len = target.shape[1]
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(batch_size * self.n_heads, tgt_len, tgt_len).to(target.device)
        
        if grad:
            output = self.t2t_model(input, target, tgt_mask=tgt_mask)
            output = self.t2t_out(output)
        else:
            with torch.no_grad():
                output = self.t2t_model(input, target, tgt_mask=tgt_mask)
                output = self.t2t_out(output)
        
        loss = self.loss(output, target)
        
        return output, loss
    
    def decode(self):
        pass
    