import argparse
import os
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from common import config
from data import make_data_loader, make_target_data_loader
from model.sync_batchnorm.replicate import patch_replication_callback
#from model.deeplab import *
from unet import UNet
from utils.loss import MSELoss
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.func import bce_loss, prob_2_entropy, flip
from utils.summaries import TensorboardSummary
import json
import torch

class Trainer(object):
    def __init__(self, config, args):
        self.args = args
        self.config = config
        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(config)

        # Define network
        #self.model = DeepLab(num_classes=self.nclass,
        #                backbone=config.backbone,
        #                output_stride=config.out_stride,
        #                sync_bn=config.sync_bn,
        #                freeze_bn=config.freeze_bn)
        self.model = UNet(n_channels=1, n_classes=3, bilinear=True)

        #train_params = [{'params': self.model.get_1x_lr_params(), 'lr': config.lr},
        #                {'params': self.model.get_10x_lr_params(), 'lr': config.lr * config.lr_ratio}]

        # Define Optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        # Define Criterion
        # whether to use class balanced weights
        self.criterion = MSELoss(cuda=args.cuda)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(config.lr_scheduler, config.lr,
                                      config.epochs, len(self.train_loader),
                                      config.lr_step, config.warmup_epochs)
        self.summary = TensorboardSummary('./train_log')

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            # cudnn.benchmark = True
            self.model = self.model.cuda()

        self.best_pred_source = 0.0
        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint, map_location=torch.device('cpu'))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        if config.freeze_bn:
            self.model.module.freeze_bn()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            itr = epoch * len(self.train_loader) + i
            self.summary.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], itr)
            A_image, A_target = sample['image'], sample['label']

            if self.args.cuda:
                A_image, A_target = A_image.cuda(), A_target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred_source, 0., self.config.lr_ratio)

            A_output = self.model(A_image)

            self.optimizer.zero_grad()

            # Train seg network
            # Supervised loss
            #seg_loss = self.criterion(A_output, A_target)
            main_loss = self.criterion(A_output, A_target)

            main_loss.backward()

            self.optimizer.step()


            train_loss += main_loss.item()
            self.summary.writer.add_scalar('Train/MSELoss', main_loss.item(), itr)
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            # Show the results of the last iteration
            #if i == len(self.train_loader)-1:
        print("Add Train images at epoch"+str(epoch))
        self.summary.visualize_image('Train', self.config.dataset, A_image, A_target, A_output, epoch, 5)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.config.batch_size + A_image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        def get_metrics(tbar):
            test_loss = 0.0
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']

                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                with torch.no_grad():
                    output = self.model(image)


                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()

                target_ = target.cpu().numpy()
            print("Add Validation-Source images at epoch"+str(epoch))
            self.summary.visualize_image('Val', self.config.dataset, image, target, output, epoch, 5)
            print('Loss: %.3f' % test_loss)

            return test_loss

        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        err = get_metrics(tbar)

        new_pred_source = err

        if new_pred_source > self.best_pred_source:
            is_best = True
            self.best_pred_source = max(new_pred_source, self.best_pred_source)
        print('Saving state, epoch:', epoch)
        torch.save(self.model.module.state_dict(), self.args.save_folder + 'models/'
                    + 'epoch' + str(epoch) + '.pth')
        loss_file = {'err': err}
        with open(os.path.join(self.args.save_folder, 'eval', 'epoch' + str(epoch) + '.json'), 'w') as f:
            json.dump(loss_file, f)


def main():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--save_folder', default='train_log/',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    if not os.path.exists('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1]):
        os.mkdir('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1])
        os.symlink('/usr/xtmp/satellite/train_models/' + os.getcwd().split('/')[-1], args.save_folder[:-1])
        print('Create soft link!')
    if not os.path.exists(args.save_folder + 'eval/'):
        os.mkdir(args.save_folder + 'eval/')
    if not os.path.exists(args.save_folder + 'models/'):
        os.mkdir(args.save_folder + 'models/')

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(config, args)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.config.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.config.epochs):
        trainer.training(epoch)
        # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        trainer.validation(epoch)


if __name__ == "__main__":
    main()
