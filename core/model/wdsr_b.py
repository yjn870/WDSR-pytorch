from torch import nn
from torch.nn.utils import weight_norm
from .common import ShiftMean


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.8):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class WDSR_B(nn.Module):
    def __init__(self, args):
        super(WDSR_B, self).__init__()
        head = [weight_norm(nn.Conv2d(3, args.n_feats, kernel_size=3, padding=1))]
        body = [ResBlock(args.n_feats, args.expansion_ratio, args.res_scale, args.low_rank_ratio)
                for _ in range(args.n_res_blocks)]
        tail = [weight_norm(nn.Conv2d(args.n_feats, 3 * (args.scale ** 2), kernel_size=3, padding=1)),
                nn.PixelShuffle(args.scale)]
        skip = [weight_norm(nn.Conv2d(3, 3 * (args.scale ** 2), kernel_size=5, padding=2)), nn.PixelShuffle(args.scale)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        self.subtract_mean = args.subtract_mean
        self.shift_mean = ShiftMean(args.rgb_mean)

    def forward(self, x):
        if self.subtract_mean:
            x = self.shift_mean(x, mode='sub')

        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s

        if self.subtract_mean:
            x = self.shift_mean(x, mode='add')

        return x
