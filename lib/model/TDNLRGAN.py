import torch.nn as nn
import torch.nn.functional as F
import torch
from .blocks import NLBlockND, DilatedNLBlockND, DilatedKronNLBlockND


class TDNLRGAN_G(nn.Module):
    def __init__(self, in_channels, out_channels, temporal=False, temporal_sequence=0):
        super(TDNLRGAN_G, self).__init__()
        self.temporal = temporal
        self.temporal_sequence = temporal_sequence
        if self.temporal:
            assert temporal_sequence > 0
        # self._initialize_weights()
         # current designed for temporal
        if self.temporal:
            self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,2,2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))
            self.bn2 = nn.BatchNorm3d(128, 0.8)

            self.conv3 = nn.Conv3d(128, 256, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))
            self.bn3 = nn.BatchNorm3d(256, 0.8)

            self.conv4 = nn.Conv3d(256, 512, kernel_size=(1,4,4), padding=(0,1,1), stride=(1,2,2))
            self.bn4 = nn.BatchNorm3d(512, 0.8)

            self.nl_block = DilatedNLBlockND(in_channels=512, mode='embedded', dimension=3, bn_layer=True)
            self.down = nn.Conv3d(512, 512, kernel_size=(temporal_sequence, 3,3), padding=(0,1,1), stride=(1,1,1))
            self.bndown = nn.BatchNorm2d(512, 0.8)

            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
            self.relu = nn.ReLU()

            # self.deconv5 = nn.ConvTranspose3d(512, 256, kernel_size=(temporal_sequence, 4,4), padding=(0,1,1), stride=(1,2,2))
            # self.bn5 = nn.BatchNorm2d(256, 0.8)
            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)
            self.bn5 = nn.BatchNorm2d(256, 0.8)

            self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
            self.bn6 = nn.BatchNorm2d(128, 0.8)

            self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
            self.bn7 = nn.BatchNorm2d(64, 0.8)

            self.deconv8 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, padding=1, stride=2)
        else: 
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
            self.bn2 = nn.BatchNorm2d(128, 0.8)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
            self.bn3 = nn.BatchNorm2d(256, 0.8)

            self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2)
            self.bn4 = nn.BatchNorm2d(512, 0.8)


            self.nl_block = DilatedNLBlockND(in_channels=512, mode='embedded', dimension=2, bn_layer=True)

            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
            self.relu = nn.ReLU()

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)
            self.bn5 = nn.BatchNorm2d(256, 0.8)

            self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
            self.bn6 = nn.BatchNorm2d(128, 0.8)

            self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
            self.bn7 = nn.BatchNorm2d(64, 0.8)

            self.deconv8 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, padding=1, stride=2)   

    def forward(self, x):

        if self.temporal:
            out1 = self.conv1(x)
            # print("out1: ",out1.shape)
            out2 = self.bn2(self.conv2(self.leaky_relu(out1)))
            # print("out2: ",out2.shape)
            out3 = self.bn3(self.conv3(self.leaky_relu(out2)))
            
            out = self.bn4(self.conv4(self.leaky_relu(out3)))
            # print("out4: ",out.shape)
            # non local block
            out = self.nl_block(out)
            # print("outNL: ",out.shape)
            out = self.down(out)
            # print("outDown: ", out.shape)
            out = torch.squeeze(out,2)
            # print("outsq: ",out.shape)
            out = self.deconv5(self.relu(out))
            # print("outd5: ",out.shape)
            # print("tseq: ",self.temporal_sequence)
            # exit()
            out = self.relu(self.bn5(out))
            # out = torch.cat([out, out3], dim=1)
            out = out + out3[:,:,self.temporal_sequence//2,:,:]
            out = self.relu(self.bn6(self.deconv6(out)))
            # out = torch.cat([out, out2], dim=1)
            out = out + out2[:,:,self.temporal_sequence//2,:,:]
            out = self.relu(self.bn7(self.deconv7(out)))
            # out = torch.cat([out, out1], dim=1)
            out = out + out1[:,:,self.temporal_sequence//2,:,:]
            out = self.deconv8(out)

        else:
            out1 = self.conv1(x)
            out2 = self.bn2(self.conv2(self.leaky_relu(out1)))
            out3 = self.bn3(self.conv3(self.leaky_relu(out2)))
            out = self.bn4(self.conv4(self.leaky_relu(out3)))
            # non local block
            out = self.nl_block(out)
            out = self.relu(self.bn5(self.deconv5(self.relu(out))))
            # out = torch.cat([out, out3], dim=1)
            out = out + out3
            out = self.relu(self.bn6(self.deconv6(out)))
            # out = torch.cat([out, out2], dim=1)
            out = out + out2
            out = self.relu(self.bn7(self.deconv7(out)))
            # out = torch.cat([out, out1], dim=1)
            out = out + out1
            out = self.deconv8(out)
        out = out + x
        return out



class TDNLRGAN_D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None,img_size=128):
        super(TDNLRGAN_D, self).__init__()

        # def discriminator_block(in_filters, out_filters, bn=True):
        #     block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, 0.8))
        #     return block

        # self.model = nn.Sequential(
        #     *discriminator_block(in_channels, 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128),
        # )

        # # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        print(ds_size)
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=4,stride=2,padding=1),
                    nn.BatchNorm2d(out_filters,0.8), nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4,stride=1,padding=0)
        self.fc = nn.Linear(out_channels * 841, 1)
        self.sig = nn.Sigmoid()

    def forward(self, img):
        # out = self.model(img)
        # out = out.view(out.shape[0], -1)
        # validity = self.adv_layer(out)

        out = self.model(img)
        out = self.conv5(out)
        out = out.view(out.shape[0], -1)
        validity = self.sig(self.fc(out))

        return validity