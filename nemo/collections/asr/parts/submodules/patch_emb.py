import torch
import torch.nn as nn


class PatchEmb(nn.Module):
    def __init__(
        self,
        patch_size=(1, 4, 4),
        d_model=512,
        ):
        super(PatchEmb, self).__init__()
        
        self.patch_size = patch_size
        
    def patchify(self, x):
        n, c, h, w = x.shape
        n_patches = w // self.patch_size[2]
        
        
        