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
        patches = torch.zeros(n, n_patches, h * w * c // n_patches)
        
        for idx, x in enumerate(x):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = x[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
        