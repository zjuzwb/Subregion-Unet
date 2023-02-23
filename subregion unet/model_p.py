import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.models.utils import unetConv2, unetUp,unetUpadd,unetUp_in
import torch.nn.init as init


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class unetConvo(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, groups=4):
        super(unetConvo, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1,  groups=groups, padding_mode='reflect', stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            # self.conv1_1 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [3, 1], padding=[1,0], groups = groups, padding_mode='reflect', stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            # )
            # self.conv1_2 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [1, 1], stride=1, groups=groups), nn.BatchNorm2d(out_size), nn.ReLU()
            # )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, padding_mode='reflect', groups=groups, stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            # self.conv2_1 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [3, 1], padding=[1, 0], padding_mode='reflect', groups=groups, stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            # )
            # self.conv2_2 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [1, 1], stride=1, groups=groups), nn.BatchNorm2d(out_size), nn.ReLU()
            # )

            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [1,3], padding=[0,1], stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            # )
            # self.conv2_1 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [3,1], padding=[1,0], stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            # )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1, padding_mode='reflect', groups=4, stride=1),  nn.ReLU()
            )
            # self.conv1_1 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [3, 1], padding=[1, 0], padding_mode='reflect', groups=4, stride=1),  nn.ReLU()
            # )
            # self.conv1_2 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [1, 1], stride=1, groups=4), nn.ReLU()
            # )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, padding_mode='reflect', groups=4,stride=1), nn.ReLU()
            )
            # self.conv2_1 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [3, 1], padding=[1, 0], padding_mode='reflect', groups=4, stride=1), nn.ReLU()
            # )
            # self.conv2_2 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [1, 1], stride=1, groups=4), nn.ReLU()
            # )

            # self.conv2 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [1, 3], padding=[0, 1], stride=1), nn.ReLU()
            # )
            # self.conv2_1 = nn.Sequential(
            #     nn.Conv2d(out_size, out_size, [3, 1], padding=[1, 0], stride=1), nn.ReLU()
            # )
        self.conv3 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_size),
                                    nn.ReLU())


    def forward(self, inputs):
        x1 = self.conv1(inputs)
        # x2 = self.conv1_1(inputs)
        # x3 = self.conv1_2(inputs)
        out = x1 #+ x2 + x3

        x1 = self.conv2(out)
        # x2 = self.conv2_1(out)
        # x3 = self.conv2_2(out)
        outputs = x1 #+ x2 + x3
        # outputs2 = self.conv1_1(inputs)
        # outputs = self.conv3(outputs1 + outputs2)

        return outputs

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # print(outputs2.shape)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class unetConvo1(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, groups=4):
        super(unetConvo1, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1,  groups=groups, stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            )


            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, groups=groups, stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1, groups=4, stride=1),  nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, groups=4,stride=1), nn.ReLU()
            )

        self.conv3 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_size),
                                    nn.ReLU())


    def forward(self, inputs):
        x1 = self.conv1(inputs)
        # x2 = self.conv1_1(inputs)
        # x3 = self.conv1_2(inputs)
        out = x1 #+ x2 + x3

        x1 = self.conv2(out)
        # x2 = self.conv2_1(out)
        # x3 = self.conv2_2(out)
        outputs = x1 #+ x2 + x3
        # outputs2 = self.conv1_1(inputs)
        # outputs = self.conv3(outputs1 + outputs2)

        return outputs

class unetConvo11(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, groups=4):
        super(unetConvo11, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1,  groups=groups, stride=1), nn.InstanceNorm2d(out_size), nn.ReLU()
            )
            # self.conv1_1 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [3, 1], padding=[1,0], groups = groups, padding_mode='reflect', stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            # )
            # self.conv1_2 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [1, 1], stride=1, groups=groups), nn.BatchNorm2d(out_size), nn.ReLU()
            # )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, groups=groups, stride=1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, padding=1, groups=4, stride=1),  nn.ReLU()
            )
            # self.conv1_1 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [3, 1], padding=[1, 0], padding_mode='reflect', groups=4, stride=1),  nn.ReLU()
            # )
            # self.conv1_2 = nn.Sequential(
            #     nn.Conv2d(in_size, out_size, [1, 1], stride=1, groups=4), nn.ReLU()
            # )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, padding=1, groups=4,stride=1), nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)

        outputs = self.conv2(outputs)


        return outputs

class fianlModel(nn.Module):
    def __init__(self, inchannel=16, nclass=3):
        super(fianlModel, self).__init__()
        self.scale = [2, 2] #subregions
        self.rtio = 4  #multiplier for number of channels in the local information stream
        self.group = 4 #corresponding groups
        self.globalrtio1 = 1 # multiplier for number of channels in the global information stream
        self.globalblock = unetConvo11(inchannel, 32*self.globalrtio1, groups=1, is_batchnorm=True)
        self.block1 = unetConvo11(inchannel*self.rtio, 32*self.rtio, groups=self.group, is_batchnorm=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.block2 = unetConvo1(32*self.rtio, 64*self.rtio, groups=self.group, is_batchnorm=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.block3 = unetConvo1(64*self.rtio, 128*self.rtio, groups=self.group, is_batchnorm=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.block4 = unetConvo1(128*self.rtio, 256*self.rtio, groups=self.group, is_batchnorm=True)

        self.up1 = unetUp(256 , 128, is_deconv=True) # lnet 0.77

        self.up2 = unetUp(128, 64, is_deconv=True)
        self.up3 = unetUp(64, 32, is_deconv=True)

        self.conv_1x1 = nn.Conv2d(64, nclass, 1)
        # self.conv_1x1_a = nn.Conv2d(64, 2, 1)

    def forward(self, input):
        # input0=input
        globalout = self.globalblock(input)
        region_input = self.pixelshuffle_invert(input, self.scale)

        out1 = self.block1(region_input)
        out1m = self.maxpool1(out1)
        out2 = self.block2(out1m)
        out2m = self.maxpool2(out2)
        out3 = self.block3(out2m)
        out3m = self.maxpool3(out3)
        out4 = self.block4(out3m)
        #
        out1 = self.pixelshuffle(out1, self.scale)
        out2 = self.pixelshuffle(out2, self.scale)
        out3 = self.pixelshuffle(out3, self.scale)
        out4 = self.pixelshuffle(out4, self.scale)

        up3 = self.up1(out3, out4)
        up2 = self.up2(out2, up3)
        up1 = self.up3(out1, up2)
        up1 = torch.cat([up1, globalout], dim=1)
        out = self.conv_1x1(up1)
        return out
        # return out , out1

    def pixelshuffle(self, x, factor_hw):
        pH = factor_hw[0]
        pW = factor_hw[1]
        y = x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC // (pH * pW), iH * pH, iW * pW
        y = y.reshape(B, oC, pH, pW, iH, iW)
        y = y.permute(0, 1, 4, 2, 5, 3)  # B, oC, iH, pH, iW, pW
        y = y.reshape(B, oC, oH, oW)
        return y

    def pixelshuffle_invert(self, x, factor_hw):
        pH = factor_hw[0]
        pW = factor_hw[1]
        y = x
        B, iC, iH, iW = y.shape
        oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
        y = y.reshape(B, iC, oH, pH, oW, pW)
        y = y.permute(0, 1, 3, 5, 2, 4)  # B, iC, pH, pW, oH, oW
        y = y.reshape(B, oC, oH, oW)
        return y



if __name__=='__main__':
    print(1)
    model = fianlModel(nclass=4)
    for m in model.modules():
        if isinstance(m, nn.Conv2d)|isinstance(m, nn.ConvTranspose2d):
            print(m)
            # if m is self.con001:
            #     init.normal_(m.weight, mean=0.0, std=0.01)
            # else:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)


