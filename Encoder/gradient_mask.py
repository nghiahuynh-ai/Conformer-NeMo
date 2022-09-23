import torch
import numpy as np


class GradientMask(torch.nn.Module):

    def __init__(self, mask_ratio=0.2, mask_value=0.0):
        super().__init__()
        
        self.ratio = mask_ratio
        self.mask_value = mask_value
    
    def forward(self, input_spec):
        batch, freq, time = input_spec.shape
        num_masks = int(self.ratio * time)
        for batch_idx in range(batch):
            masked_idx = np.random.choice([i for i in range(time)], size=num_masks, replace=False)
            for idx in masked_idx:
                input_spec[batch_idx, : , idx] = self.mask_value
        return input_spec