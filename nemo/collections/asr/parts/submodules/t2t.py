import torch
import torch.nn as nn


class Text2Text(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        vocab_size=-1,
        ):
        super().__init__()
        
        self.t2t_model = nn.Transformer(
            d_model=d_model, 
            nhead=n_heads, 
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_model,
            batch_first=True,
            )
        
        self.loss = nn.CTCLoss(blank=vocab_size)
        
    def forward(self, input, target, grad=True):
        if grad:
            output = self.t2t_model(input, target)
        else:
            with torch.no_grad():
                output = self.t2t_model(input, target)
        
        logits = torch.nn.functional.log_softmax(output.transpose(1, 0), dim=-1)
        
        target_lengths = torch.tensor(target.shape[1]).long()
        logits_lengths = torch.tensor(logits.shape[0]).long()
        
        loss = self.loss(logits, logits_lengths, target, target_lengths)
        
        return output, loss
    
    def decode(self):
        pass
    