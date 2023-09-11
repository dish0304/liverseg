# python -m visdom.server

import argparse
import logging
import sys
import os
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
import random
import shutil
import time
import timm
from tqdm import tqdm


from utils.recode_print import Logger
from utils.lr_scheduler import LR_Scheduler
from utils.utils import get_dice
from utils.all_loss import DiceMeanLoss,DiceLoss

from dataloaders import make_data_loader

from model.mynet import DAS_UNet



def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser(description='Train the Net on images and target masks')

    # general config
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--ngpu', default='0', type=str)
    parser.add_argument('--epochs', default=100, type=int,help='number of total epochs to run (default: 300)',dest='epochs')

    # dataset config
    parser.add_argument('--dataset', type=str, default='lits',
                        choices=('lits','3dircadb','chaos','sliver07'))
    parser.add_argument('--batch_size', default=10, type=int)

    # network config
    parser.add_argument('--model', default='mynet', type=str)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--n_class', default=2, type=int)


    # optimizer config
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'])
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,help='initial learning rate (default:5e-4)',dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,help='weight decay (default: 1e-5)')

    # save configs
    parser.add_argument('--log_dir', default='./log/train/')
    parser.add_argument('--save', default='./checkpoint/')



    return parser.parse_args()

def main():
    # init args #
    args=get_args()
    logging.info(str(args))

    # creat model save path
    dir_checkpoint =args.save+ args.model+'/'+args.dataset+'_'+args.plan+'/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    #writer
    log_dir = args.log_dir+'/'+args.model+'/'+args.dataset+'_'+args.plan \
              +'_LR_'+str(args.lr)+'_BS_'+str(args.batch_size)+'/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    path_log='./log/' + args.model + '/'
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    sys.stdout = Logger(path_log +args.model + '_' + args.dataset + '_' + args.plan + '.txt')
    print('------------------ New Start ------------------')
    start_time=time.time()
    print('Start time is ',time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time)))
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed=2021
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    print(str(args))

    # building  network #
    logging.info('--- building network ---')
    print('--- building network ---')

    if args.model=='mynet':
        net=MyNet(n_channels=1,n_classes=args.n_class)
    elif args.model=='mynet_3d':
        net=MyNet_3D(n_channels=1,n_classes=args.n_class)


    else:
        raise(NotImplementedError('model {} not implement'.format(args.model)))

    net.to(device=device)
    n_params = sum([p.data.nelement() for p in net.parameters()])
    print('--- total parameters = {} ---'.format(n_params))

    # prepare data #
    print('--- loading dataset ---')
    kwargs = {'num_workers': 20, 'pin_memory': True}
    train_loader, val_loader, test_loader = make_data_loader(args, **kwargs)

    # optimizer & loss  #
    print('--- configing optimizer & losses ---')
    lr = args.lr
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999),eps=1e-08,weight_decay=1e-8)
    print('optimizer:',optimizer)
    lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))


    #可用loss函数
    loss_fn = {}
    loss_fn['cross_loss'] = nn.CrossEntropyLoss()
    loss_fn['dice_loss'] = DiceMeanLoss()
    loss_fn['dice_single_loss'] = DiceLoss()

    #训练时所选loss
    criterion = loss_fn['cross_loss']
    print('loss function:',criterion)

    #   strat training  #
    print('--- start training ---')

    best_pre=300

    for epoch in range(1, args.epochs + 1):
        print('\n=>Epoches %i, learning rate = %.7f' % (epoch, optimizer.param_groups[0]['lr']))
        train(args, device, epoch, net, train_loader, optimizer, criterion, writer,lr_scheduler)
        val_loss, dice = val(args, device, epoch, net, val_loader, optimizer, criterion, writer)


        print('Val loss is:{}'.format(val_loss))
        print('Val Dice for class_1 is:{}'.format(dice))

        is_best = False
        if val_loss < best_pre:
            is_best = True
            best_pre = val_loss
        save_checkpoint({'epoch': epoch,
                         # 'state_dict': net.module.state_dict(),# 多个GPU 'state_dict': net.module.state_dict()
                         'state_dict': net.state_dict(),  # 单个GPU 'state_dict': net.state_dict()
                         'best_pre': best_pre},
                        is_best,
                        dir_checkpoint,
                        args.model)
    writer.close()
    end_time = time.time()
    print('End time is ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    total_time=end_time-start_time
    print('Total run time is {} days {} hours {} minutes {} seconds.'.format(int(total_time/(3600*24)),int(total_time/3600),int(total_time%3600/60),int(total_time%3600%60)))


def train(args, device,epoch,net, train_loader,optimizer, criterion, writer,lr_scheduler=None):
    net.train()
    n_train = len(train_loader) * args.batch_size
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
        for batch_idx,sample in enumerate(train_loader):
            image, target = sample['image'], sample['target']
            # print('image:',image.shape)
            # print('target:',image.shape)

            image = image.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if args.n_class == 1 else torch.long
            seg_pred = net(image)

            target = target.to(device=device, dtype=mask_type)

            loss = criterion(seg_pred, target)


            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            # backward
            lr = lr_scheduler(optimizer, batch_idx, epoch, 0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            global_step=(epoch-1)*len(train_loader)+batch_idx
            writer.add_scalar('train_loss', loss, global_step)
            writer.add_scalar('lr', lr,global_step)

            pbar.update(image.shape[0])
    writer.add_scalar('train_epoch_loss', float(epoch_loss), epoch)

def val(args, device,epoch, net, val_loader,optimizer, criterion, writer):
    net.eval()
    n_val = len(val_loader) * args.batch_size
    dice_list=[]
    epoch_loss=[]
    with torch.no_grad():
        with tqdm(total=n_val, desc=f'Val loss: ', unit='img') as pbar:
            for batch_idx,sample in enumerate(val_loader):
                image, target = sample['image'], sample['target']

                image = image.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                target = target.to(device=device, dtype=mask_type)

                # forward
                seg_pred = net(image)

                # calculate cross_loss
                if len(seg_pred.shape)>2 and len(target.shape)>2:
                    mask=target
                else:
                    mask = target.view(-1)

                loss = criterion(seg_pred, mask)
                epoch_loss.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(image.shape[0])
                writer.add_scalar('val_loss', loss, (epoch-1) * len(val_loader) + batch_idx)

                # calculate dice
                if net.n_classes>1:
                    seg_pred = F.softmax(seg_pred, dim=1)
                    seg_pred = torch.argmax(seg_pred, axis=1)
                    seg_pred = seg_pred.squeeze().cpu().numpy()


                    target = target.squeeze().cpu().numpy()

                    seg_c = np.uint(seg_pred == 1)
                    target_c = np.uint(target == 1)
                else:
                    seg_pred = seg_pred.squeeze().cpu().numpy()
                    target = target.cpu().numpy()
                    seg_c = np.uint(seg_pred >0.5)
                    target_c = np.uint(target >= 1)



                if seg_c.sum() > 0 or target_c.sum() > 0:
                    dice_1 = get_dice(seg_c, target_c)
                    dice_list.append(dice_1)


            dice_ave=np.mean(dice_list)

        writer.add_scalar('Dice_for_class_1/val_epoch', dice_ave,epoch)
        return np.mean(epoch_loss),dice_ave


def save_checkpoint(state, is_best, path, arch, filename='checkpoint.pth.tar'):
    filename = 'checkpoint.pth.tar'
    prefix_save = os.path.join(path, arch)
    checkpoint_name = prefix_save + '_' + filename
    torch.save(state, checkpoint_name)

    if is_best:
        shutil.copyfile(checkpoint_name, prefix_save + '_model_best.pth.tar')

if __name__ == '__main__':
    main()