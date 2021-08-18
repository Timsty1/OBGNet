import torch
import torch.nn as nn
from torchvision import models
from net.B2_ResNet50 import B2_ResNet
from net.HolisticAttention import HA
import math


class OcclusionExtractionModule(nn.Module):
    def __init__(self):
        super(OcclusionExtractionModule, self).__init__()

        self.u_925 = EpipolarREB(3, 16, decay=4, step=2)
        self.u_sd1 = SpatialREB(16, 32, step=3)
        self.u_523 = EpipolarREB(32, 32, decay=2, step=3)
        self.u_sd2 = SpatialREB(32, 32, step=3)
        self.u_321 = EpipolarREB(32, 32, decay=2, step=3)

        self.v_925 = EpipolarREB(3, 16, type='v', decay=4, step=2)
        self.v_sd1 = SpatialREB(16, 32, step=3)
        self.v_523 = EpipolarREB(32, 32, type='v', decay=2, step=3)
        self.v_sd2 = SpatialREB(32, 32, step=3)
        self.v_321 = EpipolarREB(32, 32, type='v', decay=2, step=3)

        self.act = nn.ReLU()

    def forward(self, u, v):
        u = u.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, n
        v = v.permute(0, 2, 3, 4, 1).contiguous()

        u_fea = self.u_321(self.u_sd2(self.u_523(self.u_sd1(self.u_925(u))))).squeeze(4)
        v_fea = self.v_321(self.v_sd2(self.v_523(self.v_sd1(self.v_925(v))))).squeeze(4)

        return u_fea, v_fea


class EpipolarREB(nn.Module):
    def __init__(self, in_c, out_c, type='u', decay=2, step=2):
        super(EpipolarREB, self).__init__()
        if decay == 2:
            ks = 3
        elif decay == 4:
            ks = 5
        m = []
        if type == 'u':
            self.downsample = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, ks), padding=(0, 1, 0))
            for i in range(step):
                m.append(EpipolarIRB(out_c, out_c, type='u'))
            self.encoder = nn.Sequential(*m)
        elif type == 'v':
            self.downsample = nn.Conv3d(in_c, out_c, kernel_size=(3, 1, ks), padding=(1, 0, 0))
            for i in range(step):
                m.append(EpipolarIRB(out_c, out_c, type='u'))
            self.encoder = nn.Sequential(*m)

    def forward(self, x):
        return self.encoder(self.downsample(x))


class SpatialREB(nn.Module):
    def __init__(self, in_c, out_c, step=2):
        super(SpatialREB, self).__init__()
        stride = 2
        m = []
        self.downsample = nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=(stride, stride, 1))
        for i in range(step):
            m.append(SpatialIRB(out_c, out_c))
        self.encoder = nn.Sequential(*m)

    def forward(self, x):
        return self.encoder(self.downsample(x))


class EpipolarIRB(nn.Module):
    def __init__(self, in_c, out_c, type='u'):
        super(EpipolarIRB, self).__init__()
        self.trans = None
        if in_c != out_c:
            self.trans = nn.Conv3d(in_c, out_c, kernel_size=1)
        if type == 'u':
            self.b1_conv1 = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            self.b1_conv2 = nn.Conv3d(out_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        elif type == 'v':
            self.b1_conv1 = nn.Conv3d(in_c, out_c, kernel_size=(3, 1, 3), padding=(1, 0, 1))
            self.b1_conv2 = nn.Conv3d(out_c, out_c, kernel_size=(3, 1, 3), padding=(1, 0, 1))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(out_c)
        self.bn2 = nn.BatchNorm3d(out_c)

    def forward(self, x):
        fea = self.bn2(self.b1_conv2(self.relu(self.bn1(self.b1_conv1(x)))))
        if self.trans != None:
            x = self.trans(x)
        return self.relu(fea + x)


class SpatialIRB(nn.Module):
    def __init__(self, in_c, out_c):
        super(SpatialIRB, self).__init__()
        self.trans = None
        if in_c != out_c:
            self.trans = nn.Conv3d(in_c, out_c, kernel_size=1)

        self.b1_conv1 = nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), padding=(1, 1, 0))  # 333?
        self.b1_conv2 = nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(out_c)
        self.bn2 = nn.BatchNorm3d(out_c)

    def forward(self, x):
        fea = self.bn2(self.b1_conv2(self.relu(self.bn1(self.b1_conv1(x)))))
        if self.trans != None:
            x = self.trans(x)
        return self.relu(fea + x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, edge=False):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

        self.edge = edge
        if edge:
            self.edge_pre = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        sa_pre = self.conv5(x)

        if self.edge:
            return sa_pre, self.edge_pre(x)
        else:
            return sa_pre


class OBGNet(nn.Module):
    def __init__(self, pretrained=True):
        super(OBGNet, self).__init__()
        self.pretrained = pretrained

        # EPI part
        self.mvfem = OcclusionExtractionModule()
        self.epi_self_refine_1 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # CVI part
        self.resnet = B2_ResNet()
        self.c2_1rfb = RFB(512, 64)
        self.c3_1rfb = RFB(1024, 64)
        self.c4_1rfb = RFB(2048, 64)
        self.agg1 = aggregation(64)

        self.c2_2rfb = RFB(512, 64)
        self.c3_2rfb = RFB(1024, 64)
        self.c4_2rfb = RFB(2048, 64)
        self.agg2 = aggregation(64, edge=False)

        self.epi_ha = HA()
        self.epi_self_refine_2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 64, kernel_size=3, padding=1)
        )

        self.cvi_sa_ha = HA()
        self.cvi_edge_ha_layer2 = HA()
        self.cvi_edge_ha_layer3 = HA()
        self.cvi_edge_ha_layer4 = HA()

        self.epi_edge_pre = nn.Conv2d(64, 1, kernel_size=1)

        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.initialize_weights()

    def forward(self, u, v, img):

        # EPI part
        epi_fea = self.epi_self_refine_1(torch.cat(self.mvfem(u, v), dim=1))  # 64

        # CVI part_1
        x = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(img)))  # 64 * 128 * 128
        x1 = self.resnet.layer1(x)  # 256 * 128 * 128
        x2 = self.resnet.layer2(x1)  # 512 * 64 * 64
        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 * 32 * 32
        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 * 16 * 16

        x2_1, x3_1, x4_1 = self.c2_1rfb(x2_1), self.c3_1rfb(x3_1), self.c4_1rfb(x4_1)
        initial_sa = self.agg1(x4_1, x3_1, x2_1)

        # S2E Guiding Flow
        atted_epi_fea = self.epi_ha(initial_sa.sigmoid(), epi_fea)
        atted_epi_fea = self.epi_self_refine_2(atted_epi_fea)

        # edge prediction
        epi_edge = self.epi_edge_pre(atted_epi_fea)

        # CVI part_2 & E2S Guiding Flows
        x2_2 = self.cvi_sa_ha(initial_sa.sigmoid(), x2) + self.cvi_edge_ha_layer2(epi_edge.sigmoid(), x2)
        x3_2 = self.resnet.layer3_2(x2_2)
        x3_2 = x3_2 + self.cvi_edge_ha_layer3(self.maxpool(epi_edge).sigmoid(), x3_2)
        x4_2 = self.resnet.layer4_2(x3_2)
        x4_2 = x4_2 + self.cvi_edge_ha_layer4(self.maxpool(self.maxpool(epi_edge)).sigmoid(), x4_2)
        x2_2, x3_2, x4_2 = self.c2_2rfb(x2_2), self.c3_2rfb(x3_2), self.c4_2rfb(x4_2)

        final_sa = self.agg2(x4_2, x3_2, x2_2)

        return self.up(epi_edge), self.up(initial_sa), self.up(final_sa)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.pretrained:
            res50 = models.resnet50(pretrained=True)
            pretrained_dict = res50.state_dict()
            all_params = {}
            for k, v in self.resnet.state_dict().items():
                if k in pretrained_dict.keys():
                    v = pretrained_dict[k]
                    all_params[k] = v
                elif '_1' in k:
                    name = k.split('_1')[0] + k.split('_1')[1]
                    v = pretrained_dict[name]
                    all_params[k] = v
                elif '_2' in k:
                    name = k.split('_2')[0] + k.split('_2')[1]
                    v = pretrained_dict[name]
                    all_params[k] = v
            assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
            self.resnet.load_state_dict(all_params)
