from __future__ import division , print_function

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from layers import L2Norm,MultiBoxLoss
from layers import Detect
from layers import PriorBox
from data.config import cfg


class FEM(nn.Module):
    """docstring for FEM"""

    def __init__(self, in_planes):
        super(FEM, self).__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv2d(
            in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
                      stride=1, padding=3, dilation=3)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = F.relu(out, inplace=True)
        return out


class DSFD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head1, head2, num_classes):
        super(DSFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        assert(num_classes == 2)
        self.cfg = cfg
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)

        # Feature Pyramid Network
        self.fpn_topdown6 = nn.Conv2d(256, 256,  kernel_size=1, stride=1, padding=0)
        self.fpn_topdown5 = nn.Conv2d(256, 512,  kernel_size=1, stride=1, padding=0)
        self.fpn_topdown4 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0)
        self.fpn_topdown3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.fpn_topdown2 = nn.Conv2d(512, 512,  kernel_size=1, stride=1, padding=0)
        self.fpn_topdown1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)


        self.fpn_latlayer5 = nn.Conv2d(512, 512,   kernel_size=1, stride=1, padding=0)
        self.fpn_latlayer4 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.fpn_latlayer3 = nn.Conv2d(512, 512,   kernel_size=1, stride=1, padding=0)
        self.fpn_latlayer2 = nn.Conv2d(512, 512,   kernel_size=1, stride=1, padding=0)
        self.fpn_latlayer1 = nn.Conv2d(256, 256,   kernel_size=1, stride=1, padding=0)

        self.fpn_topdown = nn.ModuleList([
            nn.Conv2d(256, 256, 1, 1, padding=0),
            nn.Conv2d(256, 512, 1, 1, padding=0),
            nn.Conv2d(512, 1024, 1, 1, padding=0),
            nn.Conv2d(1024, 512, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(512, 256, 1, 1, padding=0),
        ])

        self.fpn_latlayer = nn.ModuleList([
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(1024, 1024, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(512, 512, 1, 1, padding=0),
            nn.Conv2d(256, 256, 1, 1, padding=0),
        ])

        self.fpn_fem = nn.ModuleList([
            FEM(256), FEM(512), FEM(512),
            FEM(1024), FEM(512), FEM(256),
        ])

        # Feature enhance module
        fem_cfg = [256, 512, 512, 1024, 512, 256]
        self.fpn_fem3_3 = FEM(fem_cfg[0])
        self.fpn_fem4_3 = FEM(fem_cfg[1])
        self.fpn_fem5_3 = FEM(fem_cfg[2])
        self.fpn_fem7 = FEM(fem_cfg[3])
        self.fpn_fem6_2 = FEM(fem_cfg[4])
        self.fpn_fem7_2 = FEM(fem_cfg[5])

        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)

        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])

        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])

        if self.phase=='test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """    

        image_size = [x.shape[2] , x.shape[3]]
        pal1_sources = list()
        pal2_sources = list()
        loc_pal1 = list()
        conf_pal1 = list()
        loc_pal2 = list()
        conf_pal2 = list()

        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self.vgg[k](x)

        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        
        # apply vgg up to conv4_3 relu
        for k in range(16, 23):
            x = self.vgg[k](x)

        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)
        
        # apply extra layers and cache source layer outputs
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        of6 = x
        pal1_sources.append(of6)

        ## part2
        lfpn6 = self.fpn_topdown6(of6)
        lfpn5 = self._upsample_product(self.fpn_topdown5(of6), self.fpn_latlayer5(of5))
        lfpn4 = self._upsample_product(self.fpn_topdown4(of5), self.fpn_latlayer4(of4))
        lfpn3 = self._upsample_product(self.fpn_topdown3(of4), self.fpn_latlayer3(of3))
        lfpn2 = self._upsample_product(self.fpn_topdown2(of3), self.fpn_latlayer2(of2))
        lfpn1 = self._upsample_product(self.fpn_topdown1(of2), self.fpn_latlayer1(of1))


        ef1 = self.fpn_fem3_3(lfpn1)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem4_3(lfpn2)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem5_3(lfpn3)
        ef3 = self.L2Normef3(ef3)

        ef4 = self.fpn_fem7(lfpn4)
        ef5 = self.fpn_fem6_2(lfpn5)
        ef6 = self.fpn_fem7_2(lfpn6)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)

        ## first shot 
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal1.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        ## second shot
        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_pal2.append(c(x).permute(0, 2, 3, 1).contiguous())

        # first shot
        loc_pal1 = torch.cat([o.view(o.size(0), -1) for o in loc_pal1], 1)
        conf_pal1 = torch.cat([o.view(o.size(0), -1) for o in conf_pal1], 1)
        
        # second shot
        loc_pal2 = torch.cat([o.view(o.size(0), -1) for o in loc_pal2], 1)
        conf_pal2 = torch.cat([o.view(o.size(0), -1) for o in conf_pal2], 1)

        if self.phase == 'test':
            # 测试时， 仅使用shot2 的输出
            output = self.detect(
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
                                            self.num_classes)),                # conf preds
            )
        else:
            ## 训练时，使用shot1 和 shot2 的输出
            output = (
                loc_pal1.view(loc_pal1.size(0), -1, 4),
                conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
                loc_pal2.view(loc_pal2.size(0), -1, 4),
                conf_pal2.view(conf_pal2.size(0), -1, self.num_classes))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m,nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()

        if isinstance(m,nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax  = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = ( torch.cat([bmax, chunk[3]], dim=1) if len_conf==0 else torch.cat([chunk[3],bmax],dim=1) )
        if len(chunk)==6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1) 
        elif len(chunk)==8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1) 
        return cls

    def _upsample_product(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        # Deprecation warning. align_corners=False default in 0.4.0, but in 0.3.0 it was True
        # Original code was written in 0.3.1, I guess this is correct.
        return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=True) * y


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(vgg_cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in vgg_cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # add conv6, conv7
    conv6 = nn.Conv2d(512, 1024, kernel_size=3 , padding=1)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True) ]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)

def build_net_vgg(phase, num_classes=2):
    base = vgg(vgg_cfg, 3)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    return DSFD(phase, base, extras, head1, head2, num_classes)

if __name__ == '__main__':
    inputs = Variable(torch.randn(1, 3, 640, 640))
    net = build_net_vgg('train', 2)
    out = net(inputs)
    print(net)
    print(out)