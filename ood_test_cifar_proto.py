
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
parser.add_argument('--in-dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--out-dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100'])
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

    if args.in_dataset == 'cifar10' and args.out_dataset == 'cifar100':
        print("Using CIFAR 10 as in dataset")
        in_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        out_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 10
    elif args.in_dataset == 'cifar100' and args.out_dataset == 'cifar10':
        print("Using CIFAR100 as in dataset")
        in_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        out_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset', train=False, transform=test_transform)
        num_classes = 100
    else:
        raise NotImplementedError

    in_loader = torch.utils.data.DataLoader(
        in_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True
    )

    out_loader = torch.utils.data.DataLoader(
        out_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=True
    )

    backbone = WideResNet(args.layers, args.protodim, args.widen_factor, dropRate=args.droprate)
    net = ProtoWRN(backbone, num_classes, args.protodim)
    net.cuda()
    net.load_state_dict(torch.load(args.load))

    in_results = test(net, in_loader)
    out_results = test(net, out_loader)

    AUROC = sk.roc_auc_score(
        [1 for _ in range(len(in_results))] + [0 for _ in range(len(out_results))],
        in_results + out_results
    )

    print("AUROC = ", AUROC)


def test(net, test_loader):
    net.eval()
    scores = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()

            # forward
            ood_scores = net.get_ood_scores(data, target)

            for s in ood_scores:
                scores.append(s)

    return scores

if __name__ == "__main__":
    main()