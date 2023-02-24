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
        
        self.t2t_out = nn.Linear(d_model, d_model)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_value = torch.tensor(float('inf'))
        
    def forward(self, src, tgt, grad=True):
        
        # tgt = tgt.transpose(1, 0)
        tgt_input = tgt.narrow(1, 0, -1)
        tgt_expect = tgt.narrow(1, 1, tgt.size(1))
        
        tgt_mask = self.get_tgt_mask(tgt_input.shape[1]).to(tgt.device)
         
        if grad:
            output = self.t2t_model(src, tgt_input, tgt_mask=tgt_mask)
            output = self.t2t_out(output)
        else:
            with torch.no_grad():
                output = self.t2t_model(src, tgt_input, tgt_mask=tgt_mask)
                output = self.t2t_out(output)
        
        del tgt_input, tgt_mask
        
        # (B, t, D) -> (B, D, T)
        output = output.transpose(-1, -2)
        tgt_expect = tgt_expect.transpose(-1, -2)
        
        self.loss_value = self.loss_fn(output, tgt_expect)
        
        del tgt_expect
        
        # (B, D, T) -> (B, D, T-1) -> (B, T-1, D)
        output = output.narrow(-1, 0, -1).transpose(-1, -2)

        return output
    
    def get_loss(self):
        return self.loss_value
    
    def get_tgt_mask(self, tgt_size):
        tgt_mask = torch.tril(torch.ones(tgt_size, tgt_size) == 1)
        tgt_mask = tgt_mask.float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        return tgt_mask
    
    def decode(self):
        pass
    