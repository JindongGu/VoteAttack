import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rout import *

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, mode, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.mode = mode
        kernel_size = 4
        
        # DR
        if self.mode == 'DR':
            self.num_caps  = 64
            self.incaps_size = 8
            self.caps_size = 16
            self.bn = nn.BatchNorm2d(self.num_caps*self.incaps_size)
            self.rout = DynamicRouting2d(self.num_caps, num_classes, self.incaps_size, self.caps_size, kernel_size=kernel_size, padding=0)

        # Vote_Attack
        elif self.mode == 'Vote_Attack':
            self.num_caps  = 64
            self.incaps_size = 8
            self.caps_size = 16
            self.bn = nn.BatchNorm2d(self.num_caps*self.incaps_size)
            self.rout = Vote_Attack(self.num_caps, num_classes, self.incaps_size, self.caps_size, kernel_size=kernel_size, padding=0)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    

    def forward(self, x, num_heads=1):            
        out = F.relu(self.bn_1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # DR or No_R
        if self.mode == 'DR':
            pose = self.bn(out)
            b, c, h, w = pose.shape
            pose = pose.permute(0, 2, 3, 1).contiguous()
            pose = squash(pose.view(b, h, w, self.num_caps, self.incaps_size))
            pose = pose.view(b, h, w, -1)
            pose = pose.permute(0, 3, 1, 2)

            out = self.rout(pose).squeeze()
            out = out.view(b, -1, self.caps_size) 

            out = out.norm(dim=-1)
            out = out / out.sum(dim=1, keepdim=True)
            out = out.log()
            

        elif self.mode == 'Vote_Attack':
            
            pose = self.bn(out)
            b, c, h, w = pose.shape
            pose = pose.permute(0, 2, 3, 1).contiguous()
            pose = squash(pose.view(b, h, w, self.num_caps, self.incaps_size))
            pose = pose.view(b, h, w, -1)
            pose = pose.permute(0, 3, 1, 2)

            out = self.rout(pose)
            
            out = out.norm(dim=-1)
            out = out / out.sum(dim=2, keepdim=True)
            out = out.reshape(-1, 10)
            out = out.log()

        return out


def Capsnet(mode):
    return ResNet(BasicBlock, [2, 2, 2, 2], mode)
    










        





