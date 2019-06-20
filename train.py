import os
import random

from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.option import parser
from core.model import WDSR_A, WDSR_B
from core.data.div2k import DIV2K
from core.data.utils import quantize
from core.utils import AverageMeter, adjust_lr, calc_psnr, save_checkpoint


def print_information(model, args):
    print('=' * 30)
    print('model          :', args.model)
    print('parameters     :', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('scale          :', args.scale)
    print('n-feats        :', args.n_feats)
    print('n-res-blocks   :', args.n_res_blocks)
    print('expansion-ratio:', args.expansion_ratio)
    if args.model == 'WDSR-B':
        print('low-rank-ratio :', args.low_rank_ratio)
    print('res-scale      :', args.res_scale)
    print('=' * 30)
    print()


def train(dataset, loader, model, criterion, optimizer, tag=''):
    losses = AverageMeter()

    # Set the model to training mode
    model.train()

    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)

        for data in loader:
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)

            # Predict results and calculate loss
            sr = model(lr)
            loss = criterion(sr, hr)

            # Update loss
            losses.update(loss.item(), lr.shape[0])

            # Compute gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.4f}'.format(losses.avg))
            t.update(lr.shape[0])

        return losses.avg


def test(dataset, loader, model, criterion, args, tag=''):
    losses = AverageMeter()
    psnr = AverageMeter()

    # Set the model to evaluation mode
    model.eval()

    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)

        for data in loader:
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                sr = model(lr)

            loss = criterion(sr, hr)

            # Update loss
            losses.update(loss.item(), lr.shape[0])

            # Quantize results
            sr = quantize(sr, args.rgb_range)

            # Update PSNR
            psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=args.rgb_range[1]), lr.shape[0])

            t.set_postfix(loss='{:.4f}'.format(losses.avg))
            t.update(lr.shape[0])

        return losses.avg, psnr.avg


if __name__ == '__main__':
    # Define specific options and parse arguments
    parser.add_argument('--dataset-dir', type=str, required=True, help='DIV2K Dataset Root Directory')
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    # Set cuDNN auto-tuner and get device for learning
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create a directory to store weights
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create model
    if args.model == 'WDSR-B':
        model = WDSR_B(args).to(device)
    else:
        model = WDSR_A(args).to(device)

    print_information(model, args)

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)

    # Prepare dataset
    train_dataset = DIV2K(args, train=True)
    valid_dataset = DIV2K(args, train=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)

    best_epoch = 0
    best_loss = 0
    best_psnr = 0

    output_name = '{}-f{}-b{}-r{}-x{}'.format(args.model, args.n_feats, args.n_res_blocks, args.expansion_ratio, args.scale)
    checkpoint_name = '{}-latest.pth.tar'.format(output_name)

    # Set TensorBoard writer
    summary_writer = SummaryWriter(os.path.join(args.output_dir, 'logs', output_name))

    for epoch in range(args.epochs):
        # Adjust learning rate
        lr = adjust_lr(optimizer, args.lr, epoch, args.lr_decay_steps, args.lr_decay_gamma)

        print('[epoch: {}/{}]'.format(epoch + 1, args.epochs))

        ######################
        # Train
        ######################

        train_loss = train(train_dataset, train_dataloader, model, criterion, optimizer, tag='train')

        ######################
        # Validate
        ######################

        valid_loss, valid_psnr = test(valid_dataset, valid_dataloader, model, criterion, args, tag='valid')

        is_best = valid_psnr > best_psnr

        if valid_psnr > best_psnr:
            best_epoch = epoch
            best_loss = valid_loss
            best_psnr = valid_psnr

        print('* learning rate: {}'.format(lr))
        print('* PSNR: {:.4f}'.format(valid_psnr))
        print('* best PSNR: {:.4f} @ epoch: {}\n'.format(best_psnr, best_epoch + 1))

        ######################
        # Save checkpoint
        ######################

        save_checkpoint({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_psnr': valid_psnr,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.output_dir, checkpoint_name), is_best)

        ######################
        # TensorBoard
        ######################

        summary_writer.add_scalar('learning_rate', lr, epoch + 1)
        summary_writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch + 1)
        summary_writer.add_scalar('psnr', valid_psnr, epoch + 1)

    summary_writer.close()
