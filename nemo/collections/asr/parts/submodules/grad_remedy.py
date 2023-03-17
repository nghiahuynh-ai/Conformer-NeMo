import torch
import torch.nn as nn


class GradRemedy(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_asr = None
    
    def forward_asr(self, x):
        def hook(grad):
            self.grad_asr = grad
            return grad
        x.register_hook(hook)
        return x
    
    def forward_se(self, x):
        def hook(grad):
            cos_phi = (self.grad_asr * grad).sum(1) / ((self.grad_asr**2).sum(1).sqrt() * (grad**2).sum(1).sqrt())
            cos_phi_mask = (cos_phi <= 0.0).unsqueeze(-1).expand(grad.shape)
            sin_phi = torch.sqrt(1 - cos_phi**2)
            tan_theta = (grad**2).sum(1).sqrt() / (self.grad_asr**2).sum(1).sqrt()
            grad_2 = (grad**2).sum(1).sqrt()
            grad_asr_2 = (self.grad_asr**2).sum(1).sqrt().unsqueeze(-1)
            grad = cos_phi_mask * (grad + grad_2 * (sin_phi/tan_theta - cos_phi) * self.grad_asr/grad_asr_2) + ~cos_phi_mask*grad
            return grad
        x.register_hook(hook)
        return x
    
    