import torch
import numpy as np


class SpecAugment(torch.nn.Module):

    def __init__(self, freq_masks=0, time_masks=0, freq_width=10, time_width=10, mask_value=0.0):
        super(SpecAugment, self).__init__()
        
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.mask_value = mask_value

    @torch.no_grad()
    def forward(self, input_spec):
        batch, freq, time = input_spec.shape

        for batch_idx in range(batch):
            for i in range(self.freq_masks):
                x_left = np.random.randint(0, freq - self.freq_width)
                w = np.random.randint(0, self.freq_width)
                input_spec[batch_idx, x_left : x_left + w, :] = self.mask_value

            for i in range(self.time_masks):
                y_left = np.random.randint(0, time - self.time_width)
                w = np.random.randint(0, self.time_width)
                input_spec[batch_idx, :, y_left : y_left + w] = self.mask_value

        return input_spec


class SpecCutout(torch.nn.Module):

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20):
        super(SpecCutout, self).__init__()

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, input_spec):
        batch, freq, time = input_spec.shape

        for idx in range(batch):
            for i in range(self.rect_masks):
                rect_x = np.random.randint(0, freq - self.rect_freq)
                rect_y = np.random.randint(0, time - self.rect_time)
                w_x = np.random.randint(0, self.rect_freq)
                w_y = np.random.randint(0, self.rect_time)
                input_spec[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return input_spec

class SpecShot(torch.nn.Module):
    
    def __init__(self, mask_ratio=0.3, mask_value=0.0):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

    @torch.no_grad()
    def forward(self, input_spec):
        batch_size, freq_size, time_size = input_spec.shape
        prob = -1.0 * torch.rand(batch_size, freq_size, time_size) + 1.0
        mask = prob > self.mask_ratio
        input_spec = input_spec * mask.to(input_spec.device)
        return input_spec