import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.nn import functional as F
import pdb
from torchvision.models.resnet import resnet18
class SpatialTransformBlock(nn.Module):
    def __init__(self, num_classes, pooling_size, channels):
        super(SpatialTransformBlock, self).__init__()
        self.num_classes = num_classes
        self.spatial = pooling_size
        self.stn_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.stn_list.append(nn.Linear(channels, 4))
        resnet = resnet18(pretrained=True)
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='zeros')
        return x.cuda(), grid

    def transform_theta(self, theta_i, region_idx):
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,0,0] = torch.sigmoid(theta_i[:,0])
        theta[:,1,1] = torch.sigmoid(theta_i[:,1])
        theta[:,0,2] = torch.tanh(theta_i[:,2])
        theta[:,1,2] = torch.tanh(theta_i[:,3])
        theta = theta.cuda()
        return theta

    def forward(self, features):
        pred_list = []
        grid_list = list()
        bs = features.size(0)
        feature = self.base(features)
        for i in range(self.num_classes):
            stn_feature = feature       
            theta_i = self.stn_list[i](F.max_pool2d(stn_feature, stn_feature.size()[2:]).view(bs,-1)).view(-1,4)
            theta_i = self.transform_theta(theta_i, i)

            stn_i = self.stn(features, theta_i)
            pred = stn_i[0]
            pred_list.append(pred)
            grid_list.append(stn_i[1])
        return pred_list, grid_list