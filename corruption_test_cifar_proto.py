#####
# Mostly from https://github.com/rahuldesai1/TinyImageNetC/blob/master/eval_robustness.py
#####


# -*- coding: utf-8 -*-
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

parser = argparse.ArgumentParser(description='Evaluates robustness of a Cifar-X Classifier on Cifar-X-C',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'], help='Batch size.')
parser.add_argument('--protodim', type=int)

# WRN Architecture
parser.add_argument('--layers', default=16, type=int)
parser.add_argument('--widen-factor', default=4, type=int)
parser.add_argument('--droprate', default=0.3, type=float)

# Checkpoints
parser.add_argument('--load', '-l', type=str, required=True, help='Checkpoint path to resume / test.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

if args.dataset == 'cifar10':
    num_classes = 10
else:
    num_classes = 100

backbone = WideResNet(args.layers, args.protodim, args.widen_factor, dropRate=args.droprate)
net = ProtoWRN(backbone, num_classes, args.protodim)
net.cuda()
net.load_state_dict(torch.load(args.load))

if args.ngpu > 1:
    net.cuda()
if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

test_transform = trn.Compose([
    trn.ToTensor(), 
    trn.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

save_path = "/".join(args.load.split("/")[:-1]) + "/eval_results.csv"
with open(save_path, 'w+') as f:
    f.write('time(s),severity,distortion,test_loss,test_accuracy(%)\n')

class CF10CDataset(torch.utils.data.Dataset):
    def __init__(self, np_file_path, labels_file_path, transform):
        self.npy    = np.load(np_file_path)
        self.labels = np.load(labels_file_path)
        self.transform = transform

        self.labels = torch.from_numpy(self.labels).long()

    def __getitem__(self, index):
        return self.transform(self.npy[index]), self.labels[index]

    def __len__(self):
        return self.labels.shape[0]

def eval_loop(dataloader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()

            # forward
            loss, z_batch, classification_scores = net(data, target)

            # accuracy
            pred = classification_scores.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)
    return loss_avg, correct

def test_no_distortions():

    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform)
    else:
        test_data = dset.CIFAR100('/data/sauravkadavath/cifar10-dataset/', train=False, transform=test_transform, download=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True
    )

    loss_avg, correct = eval_loop(test_loader)
    return loss_avg / len(test_loader), correct / len(test_loader.dataset)

def test(distortion_name):
    ## Evaluation Code ##

    if args.dataset == 'cifar10':
        cifar_x_d_dset_path = "/data/sauravkadavath/CIFAR-10-C/"
    else:
        cifar_x_d_dset_path = "/data/sauravkadavath/CIFAR-100-C/"

    test_data = CF10CDataset(
        np_file_path=cifar_x_d_dset_path + distortion_name + ".npy",
        labels_file_path=cifar_x_d_dset_path + "labels.npy",
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)

    loss_avg, correct = eval_loop(test_loader)
    
    err = loss_avg / len(test_loader)
    acc = correct / len(test_loader.dataset)

    return err, acc 

def write_results_to_file(start_time, sev, dist, loss, accuracy):
    with open(save_path, 'a') as f:
        f.write('%05d,%s,%s,%0.5f,%0.2f\n' % (
            time.time() - start_time,
            str(sev),
            dist,
            loss,
            accuracy))

distortions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]

test_loss = []
test_accuracy = []

# Evaluation model on original Cifar-X for Reference
begin = time.time()
loss, accuracy = test_no_distortions()
write_results_to_file(begin, "N/A", "None", loss, accuracy)
print("Evaluation with no image distortions: Loss: %0.5f | Accuracy: %0.2f" % (loss, accuracy))

# Evaluate mode on Cifar-X-C
for distortion_name in distortions:
    begin = time.time()
    distortion_loss, distortion_accuracy = test(distortion_name)
    test_loss.append(distortion_loss)
    test_accuracy.append(distortion_accuracy)
    distortion_loss, distortion_accuracy = np.mean(distortion_loss), np.mean(distortion_accuracy)
    write_results_to_file(begin, "Average", distortion_name, distortion_loss, distortion_accuracy)
    print("Finished evaluation of %s: Loss: %0.5f | Accuracy: %0.2f" % (distortion_name, distortion_loss, distortion_accuracy))

print("Finished Evaluation on CIFAR-X-C. See results in %s" % save_path)
