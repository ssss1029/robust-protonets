
"""
Train a prototypical network on cifar
"""


import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet, ProtoWRN
import sklearn.metrics as sk
from PIL import Image
import itertools

parser = argparse.ArgumentParser(description='Trains CIFAR Prototypical Network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--model', '-m', type=str, default='wrn', choices=['wrn'])
parser.add_argument('--protodim', type=int)

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1)
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--test-bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', '-d', type=float, default=0.0005)

# WRN Architecture
parser.add_argument('--layers', default=16, type=int)
parser.add_argument('--widen-factor', default=4, type=int)
parser.add_argument('--droprate', default=0.3, type=float)

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./checkpoints/TEMP')

# Acceleration
parser.add_argument('--prefetch', type=int, default=4)
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

def main():

    train_transform = trn.Compose([
        trn.RandomHorizontalFlip(),
        trn.RandomCrop(32, padding=4),
        trn.ToTensor(),
        trn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == 'cifar10':
        print("Using CIFAR 10")
        train_data_in = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset', train=True, transform=train_transform)
        test_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 10
    else:
        print("Using CIFAR100")
        train_data_in = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset', train=True, transform=train_transform)
        test_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 100

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True
    )

    backbone = WideResNet(args.layers, args.protodim, args.widen_factor, dropRate=args.droprate)
    net = ProtoWRN(backbone, num_classes, args.protodim)
    net.cuda()

    optimizer = torch.optim.SGD(
        net.parameters(),
        state['learning_rate'],
        momentum=state['momentum'],
        weight_decay=state['decay'],
        nesterov=True
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate
        )
    )

    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    print('Beginning Training\n')
    with open(os.path.join(args.save, "training_log.csv"), 'w') as f:
        f.write("epoch,train_loss,test_loss,test_accuracy\n")

    # Main loop
    for epoch in range(0, args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()

        if epoch > 60:
            net.pin_prototypes(pin=True)
        else:
            net.pin_prototypes(pin=False)

        train(net, state, train_loader_in, optimizer, lr_scheduler)
        test(net, state, test_loader)

        # Save model
        torch.save(
            net.state_dict(),
            os.path.join(
                args.save,
                '{0}_{1}_layers_{2}_widenfactor_{3}_transform_epoch_{4}.pt'.format(
                    args.dataset,
                    args.model,
                    str(args.layers),
                    str(args.widen_factor),
                    str(epoch)
                )
            )
        )

        # Let us not waste space and delete the previous model
        prev_path = os.path.join(
            args.save,
            '{0}_{1}_layers_{2}_widenfactor_{3}_transform_epoch_{4}.pt'.format(
                args.dataset,
                args.model,
                str(args.layers),
                str(args.widen_factor),
                str(epoch - 1)
            )
        )

        if os.path.exists(prev_path):
            os.remove(prev_path)

        # Show results
        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
        )

        with open(os.path.join(args.save, "training_log.csv"), 'a') as f:
            f.write(f"{epoch},{state['train_loss']},{state['test_loss']},{state['test_accuracy']}\n")


def train(net, state, train_loader_in, optimizer, lr_scheduler):
    net.train()  # enter train mode
    loss_avg = 0.0
    for i, (data, targets) in enumerate(tqdm(train_loader_in)):
        data, targets = data.cuda(), targets.cuda()

        # forward
        loss, z_batch, _ = net(data, targets, print_stats=i % 100 == 0)

        # print(loss)

        # backward
        lr_scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    state['train_loss'] = loss_avg

def test(net, state, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            loss, z_batch, classification_scores = net(data, target)

            # accuracy
            pred = classification_scores.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if __name__ == "__main__":
    main()