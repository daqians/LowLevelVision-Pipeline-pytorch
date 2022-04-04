import torch.nn as nn
import torch


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ConvBlockInit(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(ConvBlockInit, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.init_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.init_conv(x)

##############################
#        U-NET-RCRN-C
##############################

def conv_inception(in_channel, out_channel, kernel, stride, padding, dilation):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False, dilation=dilation)
    )
    return layer

class MsRBlock_Down_CNN(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, growth_rate=8, bn_size=4):
        super(MsRBlock_Down_CNN, self).__init__()

        self.conv_inception1 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 1, 1)
        self.conv_inception2 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 2, 2)
        self.conv_inception3 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 3, 3)

        self.norm = nn.BatchNorm2d(3 * bn_size * growth_rate)
        self.relu = nn.ReLU(inplace=True)

        layers = [nn.BatchNorm2d(3 * bn_size * growth_rate),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(3 * bn_size * growth_rate, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_inception1(x)
        x2 = self.conv_inception2(x)
        x3 = self.conv_inception3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.model(x)

class MsRBlock_mid_CNN(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(MsRBlock_mid_CNN, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MsRBlock_Up_CNN(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, growth_rate=8, bn_size=4):
        super(MsRBlock_Up_CNN, self).__init__()
        self.conv_inception1 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 1, 1)
        self.conv_inception2 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 2, 2)
        self.conv_inception3 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 3, 3)

        layers = [
            nn.BatchNorm2d(3 * bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3 * bn_size * growth_rate, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x1 = self.conv_inception1(x)
        x2 = self.conv_inception2(x)
        x3 = self.conv_inception3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class CIRNet_CNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CIRNet_CNN, self).__init__()
        self.ini = ConvBlockInit(in_channels, 64)
        # UNet structure
        self.down1 = MsRBlock_Down_CNN(64, 128, normalize=False)
        self.down2 = MsRBlock_Down_CNN(128, 256)
        self.down3 = MsRBlock_Down_CNN(256, 512, dropout=0.5)

        self.mid = MsRBlock_mid_CNN(512, 512, normalize=False, dropout=0.5)

        self.up1 = MsRBlock_Up_CNN(512, 512, dropout=0.5)
        self.up2 = MsRBlock_Up_CNN(1024, 256)
        self.up3 = MsRBlock_Up_CNN(512, 128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        ini = self.ini(x)

        d1 = self.down1(ini)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        mid = self.mid(d3)

        u1 = self.up1(mid, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        return self.final(u3)


##############################
#        Discriminator
##############################

class discriminator_block_CNN(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, growth_rate=8, bn_size=4, normalization=True):
        super(discriminator_block_CNN, self).__init__()
        self.conv_inception1 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 1, 1)
        self.conv_inception2 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 2, 2)
        self.conv_inception3 = conv_inception(in_size, bn_size * growth_rate, 3, 1, 3, 3)

        layers = [nn.BatchNorm2d(3 * bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * bn_size * growth_rate, out_size, 4, stride=2, padding=1)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_inception1(x)
        x2 = self.conv_inception2(x)
        x3 = self.conv_inception3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, growth_rate=8, bn_size=4):
        super(Discriminator, self).__init__()

        self.growth_rate = growth_rate
        self.bn_size = bn_size

        layers = [discriminator_block_CNN(in_channels * 2, 64, normalization=False, growth_rate = self.growth_rate, bn_size = self.bn_size),
                  discriminator_block_CNN(64, 128),
                  discriminator_block_CNN(128, 256),
                  discriminator_block_CNN(256, 512),
                  nn.ZeroPad2d((1, 0, 1, 0)),
                  nn.Conv2d(512, 1, 4, padding=1, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


if __name__ == "__main__":
    model = CIRNet_CNN(
        in_channels=3,
        out_channels=3,
    ).cuda()

    x = torch.randn(1, 3, 256, 256).cuda()
    output = model(x)  # (1, 3, 256, 256)

    print(output.shape)
