import argparse
import os
import os.path as osp
import time
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist

from data.dutlfv2 import DUTLF_V2
from net.OBGNet import OBGNet
from lib.utils import weighted_f1_loss
from lib.BCE import bce2d_new

parser = argparse.ArgumentParser(description='OBGNet Training')
parser.add_argument('--root', default='/data/Timsty/dataset/DUTLF-V2', type=str, help='Path to dataset')
parser.add_argument('--workers', default=4, type=int, help='Num of workers')
parser.add_argument('--epochs', default=220, type=int, help='Max train epochs')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size of all GPUs on the current node when using Distributed Data Parallel')
parser.add_argument('--lr', default=3e-4, type=float, help='Initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='Node rank for distributed training, do not need to change')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--clip', default=0.5, type=float, help='Gradient clip')
parser.add_argument('--print_freq', default=7, type=int, help='Print frequency')
parser.add_argument('--milestones', default=[30, 60, 90, 120, 150], type=int, help='Learning rate decay steps')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre_trained model')

parser.add_argument('--cp_outdir', default='/data/Timsty/Checkpoints/OpenSourceTest/', type=str,
                    help='Checkpoints saving path')
parser.add_argument('--cpname', default='checkpoint.pth.tar', type=str,
                    help='Checkpoint suffix')
parser.add_argument('--bestcpname', default='bestM.pth.tar', type=str,
                    help='Best checkpoint suffix')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    main_worker(args.local_rank, args.nprocs, args)


def main_worker(local_rank, nprocs, args):
    best_f = 0.
    best_mae = 1.

    dist.init_process_group(backend='nccl')
    # create model
    model = OBGNet(pretrained=True)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    criterion = nn.BCEWithLogitsLoss().cuda(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    # Load Data
    dut_train = DUTLF_V2(args.root, type='train')
    dut_val = DUTLF_V2(args.root, type='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(dut_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dut_train, batch_size=args.batch_size, num_workers=args.workers,
                                               sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(dut_val)
    val_loader = torch.utils.data.DataLoader(dut_val, batch_size=args.batch_size, num_workers=args.workers,
                                             sampler=val_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # train_epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, local_rank, args)

        # record lr
        if local_rank == 0 and (epoch + 1) % 5 == 0:
            print('Current epoch:', epoch, 'Current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])

        # validate every 5 epochs, save best checkpoint
        if (epoch + 1) % 5 == 0:
            f2_avg, mae2_avg = validate(val_loader, model, criterion, local_rank, args)

            is_f_best = False
            is_mae_best = False

            if f2_avg > best_f:
                is_f_best = True
                best_f = f2_avg
            elif mae2_avg < best_mae:
                is_mae_best = True
                best_mae = mae2_avg

            if local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                    },
                    is_f_best,
                    is_mae_best,
                    outdir=args.cp_outdir,
                    filename=args.cpname,
                    bestModelName=args.bestcpname
                )


def train(train_loader, model, criterion, optimizer, scheduler, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae1s = AverageMeter('M_1', ':6.3f')
    mae2s = AverageMeter('M_2', ':6.3f')
    f1s = AverageMeter('F_1', ':6.3f')
    f2s = AverageMeter('F_2', ':6.3f')
    fEdge = AverageMeter('E_F_1', ':6.3f')
    maeEdge = AverageMeter('E_M_1', ':6.3f')
    progress = ProgressMeter(len(train_loader), [losses, mae1s, f1s, mae2s, f2s, maeEdge, fEdge],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)
        u, v, cvi = sample['u'].cuda(), sample['v'].cuda(), sample['cvi'].cuda()
        label, edge = sample['mask'].cuda(), sample['edge'].cuda()

        # compute outputs
        edge_pre, sa1, sa2 = model(u, v, cvi)
        # compute loss
        loss = bce2d_new(edge_pre, edge) + 0.3 * criterion(sa1, label) + 0.7 * criterion(sa2, label)

        # compute metrics, record loss
        mae1, f1 = metric(sa1, label)
        mae2, f2 = metric(sa2, label)
        mae_e, f_e = metric(edge_pre, edge)

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_mae1 = reduce_mean(mae1, args.nprocs)
        reduced_mae2 = reduce_mean(mae2, args.nprocs)
        reduced_maeEdge = reduce_mean(mae_e, args.nprocs)

        reduced_f1 = reduce_mean(f1, args.nprocs)
        reduced_f2 = reduce_mean(f2, args.nprocs)
        reduced_fEdge = reduce_mean(f_e, args.nprocs)

        losses.update(reduced_loss.item(), u.size(0))
        mae1s.update(reduced_mae1.item(), u.size(0))
        mae2s.update(reduced_mae2.item(), u.size(0))
        maeEdge.update(reduced_maeEdge.item(), u.size(0))

        f1s.update(reduced_f1.item(), u.size(0))
        f2s.update(reduced_f2.item(), u.size(0))
        fEdge.update(reduced_fEdge.item(), u.size(0))

        # compute gradients, clip gradients and update parameters
        loss.backward()
        clip_gradient(optimizer, args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i + 1 == len(train_loader) and local_rank == 0:
            progress.display(i + 1)

    scheduler.step()


def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mae1s = AverageMeter('M_1', ':6.3f')
    mae2s = AverageMeter('M_2', ':6.3f')
    f1s = AverageMeter('F_1', ':6.3f')
    f2s = AverageMeter('F_2', ':6.3f')

    fEdge = AverageMeter('E_F_1', ':6.3f')
    maeEdge = AverageMeter('E_M_1', ':6.3f')

    progress = ProgressMeter(len(val_loader), [losses, mae1s, f1s, mae2s, f2s, maeEdge, fEdge],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            u, v, cvi = sample['u'].cuda(), sample['v'].cuda(), sample['cvi'].cuda()
            label, edge = sample['mask'].cuda(), sample['edge'].cuda()

            # compute outputs
            edge_pre, sa1, sa2 = model(u, v, cvi)
            loss = bce2d_new(edge_pre, edge) + 0.3 * criterion(sa1, label) + 0.7 * criterion(sa2, label)

            # compute metrics, record loss
            mae1, f1 = metric(sa1, label)
            mae2, f2 = metric(sa2, label)
            mae_e, f_e = metric(edge_pre, edge)

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_mae1 = reduce_mean(mae1, args.nprocs)
            reduced_mae2 = reduce_mean(mae2, args.nprocs)
            reduced_maeEdge = reduce_mean(mae_e, args.nprocs)

            reduced_f1 = reduce_mean(f1, args.nprocs)
            reduced_f2 = reduce_mean(f2, args.nprocs)
            reduced_fEdge = reduce_mean(f_e, args.nprocs)

            losses.update(reduced_loss.item(), u.size(0))
            mae1s.update(reduced_mae1.item(), u.size(0))
            mae2s.update(reduced_mae2.item(), u.size(0))
            maeEdge.update(reduced_maeEdge.item(), u.size(0))

            f1s.update(reduced_f1.item(), u.size(0))
            f2s.update(reduced_f2.item(), u.size(0))
            fEdge.update(reduced_fEdge.item(), u.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i + 1 == len(val_loader) and local_rank == 0:
                print('Validation!')
                progress.display(i + 1)

    return f2s.avg, mae2s.avg


def save_checkpoint(state, is_f_best, is_mae_best, outdir, filename='checkpoint.pth', bestModelName='model_best.pth'):
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    torch.save(state, osp.join(outdir, filename))
    if is_f_best:
        torch.save(state, osp.join(outdir, 'f_' + bestModelName))
    if is_mae_best:
        torch.save(state, osp.join(outdir, 'mae_' + bestModelName))


class AverageMeter(object):
    '''Compute and stores the average and current value'''

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metric(outputs, label):
    with torch.no_grad():
        criterion = torch.nn.L1Loss()
        mae1 = criterion(torch.sigmoid(outputs), label)
        f1 = weighted_f1_loss(label, outputs)
        return mae1, f1


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
