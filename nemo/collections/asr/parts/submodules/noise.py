import numpy as np
import torch
import torch.nn as nn

class NoiseMixer:
    def __init__(
        self,
        real_noise_filepath=None,
        real_noise_snr=[0, 5],
        white_noise_mean=0.0,
        white_noise_std=[0.0, 0.05],
        ):
        super().__init__()
        
        self.add_noise_methods = []
        if real_noise_filepath is not None:
            self.add_noise_methods.append('add_real_noise')
            self.real_noise_corpus = np.load(real_noise_filepath, allow_pickle=True)
            self.real_noise_snr = real_noise_snr
        if white_noise_std[0] >= 0.0 and white_noise_std[1] >= white_noise_std[0]:
            # self.add_noise_methods.append('add_white_noise')
            self.white_noise_mean = white_noise_mean
            self.white_noise_std = white_noise_std
    
    def __call__(self, signal):
        method = np.random.choice(self.add_noise_methods, size=1)
        if method == 'add_real_noise':
            return self._add_real_noise(signal)
        else:
            return self._add_white_noise(signal)
    
    def _add_real_noise(self, signal):
        signal_length = signal.size(1)
  
        # extract noise from noise list
        noise = np.random.choice(self.real_noise_corpus)
        start = np.random.randint(0, len(noise) - signal_length - 1)
        noise = torch.from_numpy(noise[start:start + signal_length]).to(signal.device)
        
        # calculate power of audio and noise
        snr = torch.randint(low=self.real_noise_snr[0], high=self.real_noise_snr[1], size=(1,)).to(signal.device)
        signal_energy = torch.mean(signal**2)
        noise_energy = torch.mean(noise**2)
        coef = torch.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
        signal_coef = torch.sqrt(1 / (1 + coef**2))
        noise_coef = torch.sqrt(coef**2 / (1 + coef**2))
        signal = signal_coef * signal + noise_coef * noise
        del noise, snr, signal_energy, noise_energy, coef, signal_coef, noise_coef
        return signal
    
    def _add_white_noise(self, signal):
        std = np.random.uniform(self.white_noise_std[0], self.white_noise_std[1])
        noise = np.random.normal(self.white_noise_mean, std, size=signal.shape)
        noise = torch.from_numpy(noise).type(torch.FloatTensor)
        signal = signal + noise.to(signal.device)
        del noise
        return signal