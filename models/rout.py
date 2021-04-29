import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

import math

eps = 1e-12

# utility fuction
def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    v = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return v


class DynamicRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=1, stride=1, padding=1, iters=3):
        super(DynamicRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters
        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, pose):
        # x: [b, AC, h, w]
        b, _, h, w = pose.shape
        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        # [b, l, kkA, BD]
        pose_out = torch.matmul(self.W, pose).squeeze(-1)
        # [b, l, kkA, B, D]
        pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # [b, l, kkA, B, 1]
        b = pose.new_zeros(b, l, self.kkA, self.B, 1)
        #b = torch.cuda.FloatTensor(b, l, self.kkA, self.B, 1).normal_(mean=1./self.B, std=0.1)
        for i in range(self.iters):
            c = torch.softmax(b, dim=3) #.detach()
            s = (c * pose_out).sum(dim=2, keepdim=True)
            # [b, l, 1, B, D]
            v = squash(s)

            b = b + (v * pose_out).sum(dim=-1, keepdim=True)

        # [b, l, B, D]
        v = v.squeeze(2)
        # [b, l, BD]
        v = v.view(v.shape[0], l, -1)
        # [b, BD, l]
        v = v.transpose(1,2).contiguous()

        oh = ow = math.floor(l**(1/2))

        # [b, BD, oh, ow]
        return v.view(v.shape[0], -1, oh, ow)



class Vote_Attack(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=1, stride=1, padding=1):
        super(Vote_Attack, self).__init__()
        #num_caps, num_classes, incaps_size, caps_size
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B*D, C))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, pose):
        # x: [b, AC, h, w]
        b, _, h, w = pose.shape
        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        # [b, l, kkA, BD]
        pose_out = torch.matmul(self.W, pose).squeeze(-1)
        # [b, l, kkA, B, D]
        pose_out = pose_out.view(b, self.kkA, self.B, self.D)

        #pose_out = (pose_out*0.1).sum(dim=1, keepdim=True)
        
        pose_out = (pose_out/self.kkA).sum(dim=1, keepdim=True)

        # [b, l, 1, B, D]
        v = squash(pose_out)
        #print(s.shape, v.shape)
        return v

    

    






    


