import torch.nn as nn
import torch
from functools import partial
import torch
from torch import nn, einsum
from einops import rearrange

List = nn.ModuleList


### Helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def weights_init_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def conv_inception(in_channel, out_channel, kernel, stride, padding, dilation):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=False, dilation=dilation)
    )
    return layer


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


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, 1)
        )

    def forward(self, x):
        x = self.project_in(x)
        return self.project_out(x)


# SELF-ATTENTION with windows and skip-connections
class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, window_size=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        h, w, b = self.heads, self.window_size, x.shape[0]
        q = self.to_q(x)
        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) (x w1) (y w2) -> (b h x y) (w1 w2) c', w1=w, w2=w, h=h),
                      (q, k, v))

        # Matrix to Matrix multiplication, output self-attention reuslts
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b=b, h=h, y=x.shape[-1] // w, w1=w, w2=w)

        return self.to_out(out)


##############################
#        U-NET-RCRN-C
##############################


class MsRBlock_Down_CNN(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, growth_rate=8, bn_size=4):
        super(MsRBlock_Down_CNN, self).__init__()

        # Ensemble layer number = 1 in each block
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

        # Ensemble layer number = 1 in each block
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
#        Discriminator-CNN
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


class Discriminator_CNN(nn.Module):
    def __init__(self, in_channels=3, growth_rate=8, bn_size=4):
        super(Discriminator_CNN, self).__init__()

        self.growth_rate = growth_rate
        self.bn_size = bn_size

        layers = [discriminator_block_CNN(in_channels * 2, 64, normalization=False, growth_rate=self.growth_rate,
                                          bn_size=self.bn_size),
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


##############################
#        U-NET-RCRN-T
##############################

class TransBlock(nn.Module):
    def __init__(self, dim, depth=1, inter_dim=128, dim_head=64, heads=8, window_size=16, scale=1):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size / scale)
        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, window_size=self.window_size)),
                PreNorm(dim, FeedForward(dim, inter_dim))
            ]))
        self.outer_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            x = self.outer_conv(x) + x

        return x


class MsRBlock_Down_Trans(nn.Module):
    def __init__(self, in_size, out_size, depth=2, inter_dim=32, dim_head=64, heads=8, normalize=True, dropout=0.0):
        super(MsRBlock_Down_Trans, self).__init__()

        self.self_att1 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 1)
        self.self_att2 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 2)
        self.self_att3 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 4)

        layers = [nn.BatchNorm2d(3 * in_size),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(3 * in_size, out_size, 4, 2, 1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.self_att1(x)
        x2 = self.self_att2(x)
        x3 = self.self_att3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        return self.model(x)


class MsRBlock_mid_Trans(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(MsRBlock_mid_Trans, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MsRBlock_Up_Trans(nn.Module):
    def __init__(self, in_size, out_size, depth=2, inter_dim=32, dim_head=64, heads=8, dropout=0.0):
        super(MsRBlock_Up_Trans, self).__init__()

        self.self_att1 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 1)
        self.self_att2 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 2)
        self.self_att3 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 4)

        layers = [
            nn.BatchNorm2d(3 * in_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3 * in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x1 = self.self_att1(x)
        x2 = self.self_att2(x)
        x3 = self.self_att3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class CIRNet_Trans(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=2, dim_head=64, heads=8, inter_dim=128):
        super(CIRNet_Trans, self).__init__()
        self.ini = ConvBlockInit(in_channels, 64)
        # UNet structure
        self.down1 = MsRBlock_Down_Trans(64, 128, depth, inter_dim, dim_head, heads, normalize=False)
        self.down2 = MsRBlock_Down_Trans(128, 256, depth, inter_dim, dim_head, heads)
        self.down3 = MsRBlock_Down_Trans(256, 512, depth, inter_dim, dim_head, heads, dropout=0.5)

        self.mid = MsRBlock_mid_Trans(512, 512, normalize=False, dropout=0.5)

        self.up1 = MsRBlock_Up_Trans(512, 512, depth, inter_dim, dim_head, heads, dropout=0.5)
        self.up2 = MsRBlock_Up_Trans(1024, 256, depth, inter_dim, dim_head, heads)
        self.up3 = MsRBlock_Up_Trans(512, 128, depth, inter_dim, dim_head, heads)

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
#     Discriminator-Trans
##############################

class discriminator_block_Trans(nn.Module):
    def __init__(self, in_size, out_size, depth =2, inter_dim = 128, dim_head=64, heads=8, dropout=0.0, normalization=True):
        super(discriminator_block_Trans, self).__init__()

        self.self_att1 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 1)
        self.self_att2 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 2)
        self.self_att3 = TransBlock(in_size, depth, inter_dim, dim_head, heads, 16, 4)

        layers = [nn.BatchNorm2d(3 * in_size),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(3 * in_size, out_size, 4, 2, 1, bias=False)]

        if normalization:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.self_att1(x)
        x2 = self.self_att2(x)
        x3 = self.self_att3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = self.model(x)
        return x


class Discriminator_Trans(nn.Module):
    def __init__(self, in_channels=3, depth=2, inter_dim=128, dim_head=64, heads=8):
        super(Discriminator_Trans, self).__init__()

        layers = [discriminator_block_Trans(in_channels * 2, 64, normalization=False, depth=depth, inter_dim=inter_dim, dim_head = dim_head, heads = heads),
                  discriminator_block_Trans(64, 128, depth=depth, inter_dim=inter_dim, dim_head = dim_head, heads = heads),
                  discriminator_block_Trans(128, 256, depth=depth, inter_dim=inter_dim, dim_head = dim_head, heads = heads),
                  discriminator_block_Trans(256, 512, depth=depth, inter_dim=inter_dim, dim_head = dim_head, heads = heads),
                  nn.ZeroPad2d((1, 0, 1, 0)),
                  nn.Conv2d(512, 1, 4, padding=1, bias=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).cuda()

    # model = CIRNet_Trans(in_channels=3, out_channels=3, depth =1, inter_dim=32, dim_head=8, heads=4).cuda()
    model = CIRNet_CNN(in_channels=3, out_channels=3).cuda()
    output = model(x)  # (1, 3, 256, 256)
    print(output.shape)

    model = Discriminator_Trans(in_channels=3, depth=1, inter_dim=32, dim_head=8, heads=4).cuda()
    # model = Discriminator_CNN(in_channels=3).cuda()
    output = model(x,x)  # (1, 3, 256, 256)

    print(output.shape)
