import torch.nn as nn
import torch.nn.functional as F
import torch


class RGAN_G(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None):
        super(RGAN_G, self).__init__()

        # self.base = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
        #     nn.PReLU(),
        #     nn.Conv2d(64, 32, kernel_size=7, padding=3),
        #     nn.PReLU(),
        #     nn.Conv2d(32, 16, kernel_size=1),
        #     nn.PReLU()
        # )
        # self.last = nn.Conv2d(16, out_channels, kernel_size=5, padding=2)

        # self._initialize_weights()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128, 0.8)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(256, 0.8)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(512, 0.8)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256, 0.8)

        self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(128, 0.8)

        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.bn7 = nn.BatchNorm2d(64, 0.8)

        self.deconv8 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, padding=1, stride=2)   

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        # x = self.base(x)
        # x = self.last(x)

        out1 = self.conv1(x)
        out2 = self.bn2(self.conv2(self.leaky_relu(out1)))
        out3 = self.bn3(self.conv3(self.leaky_relu(out2)))
        out = self.bn4(self.conv4(self.leaky_relu(out3)))
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
        return out



class RGAN_D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None,img_size=128):
        super(RGAN_D, self).__init__()

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
        # ds_size = img_size // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

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
        self.fc = nn.Linear(out_channels * 5 ** 2, 1)
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