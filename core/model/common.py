import torch
from torch import nn


class ShiftMean(nn.Module):
    def __init__(self, rgb_mean):
        super(ShiftMean, self).__init__()
        self.rgb_mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.rgb_mean.to(x.device) * 255.0) / 127.5
        elif mode == 'add':
            return x * 127.5 + self.rgb_mean.to(x.device) * 255.0
        else:
            raise NotImplementedError
