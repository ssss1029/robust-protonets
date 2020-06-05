

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes=10, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.num_classes = num_classes

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=True)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class ProtoWRN(nn.Module):
    """
    ProtoWRN
    """

    def __init__(self, backbone, num_classes, protodim):
        super(ProtoWRN, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.protodim = protodim

        assert backbone.num_classes == protodim

        self.prototypes = torch.nn.Parameter(
            data = torch.zeros((num_classes, protodim)).uniform_(-10, 10),
            requires_grad = True
        )
    
    def pin_prototypes(self, pin=False):
        self.prototypes.requires_grad = not pin

    def forward(self, x, y, print_stats=False):
        if self.prototypes.requires_grad == True:
            return self.forward_2(x, y, print_stats=print_stats)
        else:
            return self.forward_1(x, y, print_stats=print_stats)

    def forward_1(self, x, y, print_stats=False):
        """
        Using L2 metrics
        """
        
        if print_stats:
            print(self.prototypes[:4, :5])

        batch_size = x.shape[0]

        # Feature embedding
        z_batch = self.backbone(x)
        
        # Determine loss for z
        loss_L2 = torch.zeros(1).cuda()
        classification_scores = torch.zeros((batch_size, self.num_classes)).cuda()
        losses_prototypes = 0
        losses_examples = 0
        for i, (z, y) in enumerate(zip(z_batch, y)):
            dists_to_prototypes = torch.sum((self.prototypes - z) ** 2, dim=1)

            # loss_prototypes = -0.1 * (torch.sum(torch.sqrt(dists_to_prototypes[:y])) + torch.sum(torch.sqrt(dists_to_prototypes[y+1:])))
            # loss_example = torch.sqrt(dists_to_prototypes[y]) * 2.0
            # losses_prototypes += loss_prototypes.item()
            # losses_examples += loss_example.item()
            # loss += loss_prototypes + loss_example
            
            loss_L2 += torch.sqrt(dists_to_prototypes[y])

            classification_scores[i] = 1 / (dists_to_prototypes + 1e-10)        

        squeeze_loss = self.squeeze_loss(x, z_batch)
        loss = squeeze_loss + loss_L2

        if print_stats:
            # print(losses_prototypes / batch_size)
            # print(losses_examples / batch_size)
            print(loss, squeeze_loss, loss_L2)
            pass

        return loss, z_batch, classification_scores

    def forward_2(self, X, targets, print_stats=False):
        """
        Using softmax loss with L2 logits as in Max-Mahalanobis Linear Discriminant Analysis Networks
        """
        
        if print_stats:
            print(self.prototypes[:4, :5])

        batch_size = X.shape[0]

        # Feature embedding
        z_batch = self.backbone(X)
        
        # Determine loss for z
        classification_scores = torch.zeros((batch_size, self.num_classes)).cuda()
        logits = torch.zeros((batch_size, self.num_classes)).cuda()
        for i, (z, _) in enumerate(zip(z_batch, targets)):
            dists_to_prototypes = torch.sum((self.prototypes - z) ** 2, dim=1)
            classification_scores[i] = 1 / (dists_to_prototypes + 1e-10)
            logits[i] = -1.0 * dists_to_prototypes
        
        loss = F.cross_entropy(logits, targets)

        if print_stats:
            print(loss)

        return loss, z_batch, classification_scores

    def get_ood_scores(self, X, targets, print_stats=False):
        
        z_batch = self.backbone(X)

        ood_scores = []
        for i, z in enumerate(z_batch):
            dists_to_prototypes = torch.sum((self.prototypes - z) ** 2, dim=1)
            ood_score = -1.0 * torch.min(dists_to_prototypes)
            ood_scores.append(ood_score.item())

        return ood_scores

    def squeeze_loss(self, X, clean_embeddings):
        
        noisy_X = X + torch.empty(X.shape).uniform_(-0.1, 0.1).cuda()
        noisy_embeddings = self.backbone(noisy_X)

        return torch.sqrt(torch.sum((noisy_embeddings - clean_embeddings) ** 2))
