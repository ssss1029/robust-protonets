
"""
Test a prototypical network on cifar
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

import attacks

parser = argparse.ArgumentParser(description='Trains CIFAR Prototypical Network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--model', '-m', type=str, default='wrn', choices=['wrn'])
parser.add_argument('--protodim', type=int)

# WRN Architecture
parser.add_argument('--layers', default=16, type=int)
parser.add_argument('--widen-factor', default=4, type=int)
parser.add_argument('--droprate', default=0.3, type=float)

# Checkpoints
parser.add_argument('--load', type=str,)
parser.add_argument('--test-bs', type=int, default=200)

# Acceleration
parser.add_argument('--prefetch', type=int, default=4)
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

def main():

    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == 'cifar10':
        print("Using CIFAR 10")
        test_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 10
    else:
        print("Using CIFAR100")
        test_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 100

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
    net.load_state_dict(torch.load(args.load))

    test(net, state, test_loader)

def test(net, state, test_loader):
    adversary = attacks.PGD_proto(epsilon=8./255, num_steps=20, step_size=2./255).cuda()

    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()

            data = adversary(net, data, target)

            # forward
            loss, z_batch, classification_scores = net(data, target)

            # accuracy
            pred = classification_scores.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

    print("Accuracy = ", state['test_accuracy'])


if __name__ == "__main__":
    main()