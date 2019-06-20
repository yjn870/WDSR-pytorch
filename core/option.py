import argparse

parser = argparse.ArgumentParser()

#################################
# Model Options
################################

parser.add_argument('--model', type=str, default='WDSR-A')  # WDSR-A, WDSR-B
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--n-feats', type=int, default=32)
parser.add_argument('--n-res-blocks', type=int, default=16)
parser.add_argument('--expansion-ratio', type=int, default=4)
parser.add_argument('--low-rank-ratio', type=float, default=0.8)
parser.add_argument('--res-scale', type=float, default=1.0)

################################
# Pre-processing, Post-processing Options
################################

parser.add_argument('--subtract-mean', type=bool, default=True)
parser.add_argument('--rgb-mean', type=list, default=[0.4488, 0.4371, 0.4040])  # Mean value of RGB in the DIV2K
parser.add_argument('--rgb-range', type=tuple, default=(0.0, 255.0))

################################
# Training Options
################################

parser.add_argument('--patch-size', type=int, default=96)  # HR Patch Size
parser.add_argument('--augment_patch', type=bool, default=True)

parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--iterations-per-epoch', type=int, default=1000)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr-decay-steps', type=int, default=200)
parser.add_argument('--lr-decay-gamma', type=float, default=0.5)

################################
# Etc Options
################################

parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=1)
