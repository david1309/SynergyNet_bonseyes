#!/usr/bin/env python3
# coding: utf-8
import os
import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# cudnn.benchmark=True  # TODO

from utils.ddfa import DDFADataset, ToTensor, Normalize, SGD_NanHandler, CenterCrop, Compose_GT, ColorJitter
from utils.ddfa import str2bool, AverageMeter
from utils.io import mkdir
from model_building import SynergyNet as SynergyNet

from data.dataloader_300wlp import dataset_from_datatool
from torch.utils.tensorboard import SummaryWriter
from plot_results import plot_results
from datetime import datetime


# global args (configuration)
args = None # define the static training setting, which wouldn't and shouldn't be changed over the whole experiements.

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('--datatool-root-dir', type=str)
    parser.add_argument('--train-tags', default="HELEN, HELEN_Flip, LFPW, LFPW_Flip", type=str)
    parser.add_argument('--val-tags', default="AFW, AFW_Flip, IBUG, IBUG_Flip", type=str)

    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--use-cuda', default='true', type=str2bool)
    parser.add_argument('--root', default='')
    parser.add_argument('--ckp-dir', default='ckpts', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--arch', default='mobilenet_v2', type=str, help="Please choose [mobilenet_v2, mobilenet_1, resnet50, resnet101, or ghostnet]")
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--img_size', default=450, type=int)
    parser.add_argument('--save-val-freq', default=10, type=int)
    parser.add_argument('--debug', default='false', type=str2bool)
    parser.add_argument('--num-lms', default=77, type=int)
    parser.add_argument('--exp-name', default="experiment", type=str)
    parser.add_argument('--crop-images', default="false", type=str2bool)
    parser.add_argument('--use-rot-inv', default=False, type=bool)

    global args
    args = parser.parse_args()

    # some other operations
    args.train_tags = [str(t) for t in args.train_tags.split(',')]
    args.val_tags = [str(t) for t in args.val_tags.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]

    now = datetime.now().strftime("%d.%m.%Y_%Hh%Mm%Ss")
    args.ckp_dir = os.path.join(args.ckp_dir, args.exp_name + "_" + now)
    args.log_file = os.path.join(args.ckp_dir, "logs", args.log_file)
    mkdir(args.ckp_dir)
    mkdir(os.path.join(args.ckp_dir, "logs"))
    mkdir(os.path.join(args.ckp_dir, "model_ckpts"))




def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

def adjust_learning_rate(optimizer, epoch, milestones=None):
    """Sets the learning rate: milestone is a list/tuple"""

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)

    n = to(epoch)

    #global lr
    lr = args.base_lr * (0.2 ** n)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    train_loader,
    model,
    optimizer,
    epoch,
    lr,
    writer,
    imgs_saving_path=None
    ):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_name = list(model.get_losses())
    losses_name.append('loss_total')
    losses_meter = [AverageMeter() for i in range(len(losses_name))]

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        losses = model(input, target)

        data_time.update(time.time() - end)

        loss_total = 0
        for j, name in enumerate(losses):
            mean_loss = losses[name].mean()
            losses_meter[j].update(mean_loss, input.size(0))
            loss_total += mean_loss

        losses_meter[j+1].update(loss_total, input.size(0))

        ### compute gradient and do SGD step
        optimizer.zero_grad()
        loss_total.backward()
        flag, _ = optimizer.step_handleNan()

        if flag:
            print("Nan encounter! Backward gradient error. Not updating the associated gradients.")

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            msg = 'Epoch: [{}][{}/{}]\t'.format(epoch, i, len(train_loader)) + \
                  'LR: {:.8f}\t'.format(lr) + \
                  'Time: {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)
            for k in range(len(losses_meter)):
                msg += '{}: {:.4f} ({:.4f})\t'.format(losses_name[k], losses_meter[k].val, losses_meter[k].avg)
            logging.info(msg)
            for k in range(len(losses_meter)):
                writer.add_scalar('TrainLoss/' + losses_name[k], losses_meter[k].val, epoch*len(train_loader) + i)

    # Plot last batch of images
    if (epoch % args.save_val_freq == 0) or (epoch==args.epochs):
        if imgs_saving_path is not None:
            n_samples = 4
            n_input = input.shape[0]
            idx = np.random.choice(n_input, n_samples, replace=False)
            input_ = input[idx]
            target_ = {k:v[idx] for k,v in target.items()}
            plot_results(
                model, 
                input_, 
                imgs_saving_path, 
                lm_with_lines=True,
                targets=target_, 
                only_gt=False
                )

def validate(
    val_loader,
    model,
    epoch,
    tot_train_samples,
    writer,
    imgs_saving_path=None
    ):
    """Network validation, and computing validation metrics"""

    # AverageMeter for statistics
    losses_name = list(model.get_losses())
    losses_name.append('loss_total')
    losses_meter = [AverageMeter() for i in range(len(losses_name))]

    model.eval()

    for i, (input, target) in enumerate(val_loader):

        with torch.no_grad():
            losses  = model(input, target)

        loss_total = 0
        for j, name in enumerate(losses):
            mean_loss = losses[name].mean()
            losses_meter[j].update(mean_loss, input.size(0))
            loss_total += mean_loss

        losses_meter[j+1].update(loss_total, input.size(0))

    # Plot last batch of images
    if imgs_saving_path is not None:
        n_samples = 4
        n_input = input.shape[0]
        idx = np.random.choice(n_input, n_samples, replace=False)
        input_ = input[idx]
        target_ = {k:v[idx] for k,v in target.items()}
        plot_results(
            model, 
            input_, 
            imgs_saving_path, 
            lm_with_lines=True,
            targets=target_, 
            only_gt=False
            )

    msg = (
        'Validation losses:\t' + \
        'Epoch: [{}]\t'.format(epoch)
    )
    for k in range(len(losses_meter)):
        msg += '{}: {:.4f} ({:.4f})\t'.format(losses_name[k], losses_meter[k].val, losses_meter[k].avg)
    logging.info(msg)

    for k in range(len(losses_meter)):
        writer.add_scalar('ValLoss/' + losses_name[k], losses_meter[k].val, tot_train_samples)

def main():
    """ Main funtion for the training process"""
    parse_args()  # parse global argsl

    # logging setup
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    print_args(args)  # print args

    # step1: define the model structure
    device = torch.device(f"cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    args.device = device
    model = SynergyNet(args)

    # step2: optimization: loss and optimization method

    optimizer = SGD_NanHandler(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # step 2.1 resume
    if args.resume:
        if Path(args.resume).is_file():
            logging.info(f'=> loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(checkpoint, strict=False)

        else:
            logging.info(f'=> no checkpoint found at {args.resume}')

    # step3: data
    # normalize = Normalize(
    #     mean=[0.498, 0.498, 0.498],
    #     std=[0.229, 0.229, 0.229]
    #     )
    # add_transforms = [normalize]
    add_transforms = []
    train_dataset = dataset_from_datatool(args.datatool_root_dir, args.train_tags, add_transforms)
    val_dataset = dataset_from_datatool(args.datatool_root_dir, args.val_tags, add_transforms)
    if args.debug:
        train_dataset.usable_annotations = train_dataset.usable_annotations[0:args.batch_size]
        val_dataset.usable_annotations = val_dataset.usable_annotations[0:args.batch_size]
    pin_memory = (args.device.type == "gpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=pin_memory, drop_last=False)

    logging.info(f"Num. training samples: {len(train_dataset)} ({len(train_loader)} batches)")
    logging.info(f"Num. validation samples: {len(val_dataset)} ({len(val_loader)} batches)")

    # step4: run
    writer = SummaryWriter(log_dir=os.path.join(args.ckp_dir, "tb_runs"))

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.milestones)

        # train for one epoch
        imgs_saving_path = os.path.join(args.ckp_dir, "images_results", "train", f"epoch_{epoch}")
        train(
            train_loader,
            model,
            optimizer,
            epoch,
            lr,
            writer,
            imgs_saving_path
        )

        # save checkpoints and current model validation
        if (epoch % args.save_val_freq == 0) or (epoch==args.epochs):

            # Validation
            tot_train_samples = (epoch + 1) * len(train_loader)
            imgs_saving_path = os.path.join(args.ckp_dir, "images_results", "val", f"epoch_{epoch}")
            validate(
                val_loader,
                model,
                epoch,
                tot_train_samples,
                writer,
                imgs_saving_path
            )

            # Checkpointing
            filename = os.path.join(args.ckp_dir, "model_ckpts", f"SynergyNet_ckp_epoch_{epoch}.pth.tar")
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                },
                filename
            )


if __name__ == '__main__':
    main()
