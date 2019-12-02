r"""Networks used in TP-GAN.

 .. _TP-GAN\: Beyond Face Rotation: Global and Local Perception GAN for Photorealistic
              and Identity Preserving Frontal View Synthesis
     https://arxiv.org/abs/1704.04086
 """
import torch
from ops import sample_truncated_normal


def _init_weight(m):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data = 0.02 * sample_truncated_normal(m.weight.shape)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    if isinstance(m, torch.nn.ConvTranspose2d):
        m.weight.data.normal_(0., 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.)
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0., 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.1)


_ACTIVATION = lambda key: (key == 'lrelu' and [torch.nn.LeakyReLU(0.2, inplace=True)]) \
    or (key == 'relu' and [torch.nn.ReLU(inplace=True)]) \
    or (key == 'tanh' and [torch.nn.Tanh()]) or (key == 'none' and [])


class _deConvBnActLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 transposed=False, output_padding=0, bn=True, activate='lrelu'):
        super(_deConvBnActLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ((not transposed) and \
             torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))\
            or torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                        output_padding=output_padding, bias=bias),
            *[torch.nn.BatchNorm2d(out_channels)][:bn],
            *_ACTIVATION(activate),
        )

    def forward(self, input):
        return self.sub_module(input)


class ResidualBlock(torch.nn.Module):
    """Residual block. """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _deConvBnActLayer(channels, channels, kernel_size, stride, padding),
            _deConvBnActLayer(channels, channels, kernel_size, stride, padding, activate='none'),
        )
        self.tail = torch.nn.LeakyReLU(0.2)

    def forward(self, input):
        return self.tail(self.sub_module(input) + input)


class DiscriminatorLocal(torch.nn.Module):
    """Discriminator network used in TP-GAN. """
    def __init__(self, num_channels=3, fmap_base=64):
        super(DiscriminatorLocal, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _deConvBnActLayer(num_channels, fmap_base, 4, 2, 1, bn=False),
            _deConvBnActLayer(fmap_base, 2 * fmap_base, 4, 2, 1),
            _deConvBnActLayer(2 * fmap_base, 4 * fmap_base, 4, 2, 1),
            _deConvBnActLayer(4 * fmap_base, 8 * fmap_base, 4, 2, 1),
            ResidualBlock(8 * fmap_base),
            _deConvBnActLayer(8 * fmap_base, 8 * fmap_base, 4, 2, 1),
            ResidualBlock(8 * fmap_base),
            torch.nn.Conv2d(8 * fmap_base, 1, 1, 1, 0, bias=True),
        )

        self.apply(_init_weight)

    def forward(self, input):
        x = self.sub_module(input)
        x = x.reshape(x.size(0), -1)

        return x


class FaceRotator(torch.nn.Module):
    """Draws frontal facial structure from the profile image. """
    def __init__(self, num_channels=3, fmap_base=64, y_dim=256, z_dim=100):
        super(FaceRotator, self).__init__()

        # encoder of global generator
        self.down0 = torch.nn.Sequential(
            _deConvBnActLayer(num_channels, fmap_base, 7, 1, 3, bn=False),
            ResidualBlock(fmap_base, 7, 1, 3),
        )
        self.down1 = torch.nn.Sequential(
            _deConvBnActLayer(fmap_base, fmap_base, 4, 2, 1),
            ResidualBlock(fmap_base, 5, 1, 2),
        )
        self.down2 = torch.nn.Sequential(
            _deConvBnActLayer(fmap_base, 2 * fmap_base, 4, 2, 1),
            ResidualBlock(2 * fmap_base),
        )
        self.down3 = torch.nn.Sequential(
            _deConvBnActLayer(2 * fmap_base, 4 * fmap_base, 4, 2, 1),
            ResidualBlock(4 * fmap_base),
        )
        self.down4 = torch.nn.Sequential(
            _deConvBnActLayer(4 * fmap_base, 8 * fmap_base, 4, 2, 1),
            ResidualBlock(8 * fmap_base),
            ResidualBlock(8 * fmap_base),
            ResidualBlock(8 * fmap_base),
            ResidualBlock(8 * fmap_base),
        )

        self.echo = torch.nn.Linear(512 * 8 * 8, 2 * y_dim, bias=True)
        self.register_buffer("ldim", torch.IntTensor([y_dim, z_dim]))

        # decoder of global generator
        self.initial = torch.nn.ModuleDict({
            '8x8': torch.nn.Sequential(
                torch.nn.Linear(y_dim + z_dim, 8 * 8 * fmap_base, bias=True),
                torch.nn.ReLU(inplace=True),),
            '32x32': _deConvBnActLayer(fmap_base, fmap_base//2, 4, 4, 0, transposed=True,
                                       bn=False, activate='relu'),
            '64x64': _deConvBnActLayer(fmap_base//2, fmap_base//4, 4, 2, 1, transposed=True,
                                       bn=False, activate='relu'),
            '128x128': _deConvBnActLayer(fmap_base//4, fmap_base//8, 4, 2, 1, transposed=True,
                                         bn=False, activate='relu')
        })

        # frontal view synthesis
        self.upscale2 = torch.nn.ModuleDict({
            '16x16': torch.nn.Sequential(
                ResidualBlock((8 + 1) * fmap_base),
                ResidualBlock((8 + 1) * fmap_base),
                ResidualBlock((8 + 1) * fmap_base),
                _deConvBnActLayer((8 + 1) * fmap_base,
                                  8 * fmap_base, 4, 2, 1, transposed=True, activate='relu'),),
            '32x32': _deConvBnActLayer((8 + 4) * fmap_base,
                                       4 * fmap_base, 4, 2, 1, transposed=True, activate='relu'),
            '64x64': _deConvBnActLayer((4 + 2) * fmap_base + 2 * num_channels + fmap_base//2,
                                       2 * fmap_base, 4, 2, 1, transposed=True, activate='relu'),
            '128x128': _deConvBnActLayer((2 + 1) * fmap_base + (2 + 1) * num_channels + fmap_base//4,
                                         fmap_base, 4, 2, 1, transposed=True, activate='relu'),
        })
        self.before_select = torch.nn.ModuleDict({
            '16x16': ResidualBlock(4 * fmap_base),
            '32x32': ResidualBlock(2 * fmap_base + 2 * num_channels + fmap_base//2),
            '64x64': ResidualBlock(fmap_base + 2 * num_channels + fmap_base//4, 5, 1, 2),
            '128x128': ResidualBlock(fmap_base + 2 * num_channels + fmap_base//8, 7, 1, 3),
        })
        self.reconstruct = torch.nn.ModuleDict({
            '16x16': torch.nn.Sequential(
                ResidualBlock((8 + 4) * fmap_base),
                ResidualBlock((8 + 4) * fmap_base),),
            '32x32': torch.nn.Sequential(
                ResidualBlock((4 + 2) * fmap_base + 2 * num_channels + fmap_base//2),
                ResidualBlock((4 + 2) * fmap_base + 2 * num_channels + fmap_base//2),),
            '64x64': torch.nn.Sequential(
                ResidualBlock((2 + 1) * fmap_base + (2 + 1) * num_channels + fmap_base//4),
                ResidualBlock((2 + 1) * fmap_base + (2 + 1) * num_channels + fmap_base//4),),
            '128x128': torch.nn.Sequential(
                ResidualBlock((1 + 1 + 1) * fmap_base + (2 + 1 + 1) * num_channels + fmap_base//8,
                              5, 1, 2),
                _deConvBnActLayer((1 + 1 + 1) * fmap_base + (2 + 1 + 1) * num_channels + fmap_base//8,
                                  fmap_base, 5, 1, 2),
                ResidualBlock(fmap_base),
                _deConvBnActLayer(fmap_base, fmap_base//2, 3, 1, 1),
            )
        })
        self.to_image = torch.nn.ModuleDict({
            '32x32': _deConvBnActLayer((4 + 2) * fmap_base + 2 * num_channels + fmap_base//2,
                                       num_channels, bn=False, activate='tanh'),
            '64x64': _deConvBnActLayer((2 + 1) * fmap_base + (2 + 1) * num_channels + fmap_base//4,
                                       num_channels, bn=False, activate='tanh'),
            '128x128': _deConvBnActLayer(fmap_base//2, num_channels, bn=False, activate='tanh'),
        })

        self.apply(_init_weight)

    def forward(self, pose_face, part_combine, check_part_combine):
        feat_128 = self.down0(pose_face)
        feat_64 = self.down1(feat_128)
        feat_32 = self.down2(feat_64)
        feat_16 = self.down3(feat_32)
        feat_8 = self.down4(feat_16)

        echo = self.echo(feat_8.reshape(feat_8.size(0), -1))
        echo = torch.max(echo[:, :self.ldim[0]], echo[:, self.ldim[0]:])

        noise = torch.normal(0., 0.02, size=(echo.size(0), self.ldim[1]), device=echo.device)
        latent = torch.cat([echo, noise], dim=1)

        initial_8 = self.initial['8x8'](latent).reshape(echo.size(0), -1, 8, 8)
        initial_32 = self.initial['32x32'](initial_8)
        initial_64 = self.initial['64x64'](initial_32)
        initial_128 = self.initial['128x128'](initial_64)

        mc = lambda left: torch.cat([left, torch.flip(left, dims=(3,))], dim=1)
        resize = lambda image, res: \
            torch.nn.functional.interpolate(image, (res, res), mode='bilinear', align_corners=False)

        up2res_16 = self.upscale2['16x16'](torch.cat([feat_8, initial_8], dim=1))
        before_select_16 = self.before_select['16x16'](feat_16)
        reconstruct_16 = self.reconstruct['16x16'](torch.cat([up2res_16, before_select_16], dim=1))

        up2res_32 = self.upscale2['32x32'](reconstruct_16)
        before_select_32 = self.before_select['32x32']\
            (torch.cat([feat_32, mc(resize(pose_face, 32)), initial_32], dim=1))
        reconstruct_32 = self.reconstruct['32x32'](torch.cat([up2res_32, before_select_32], dim=1))
        image_32 = self.to_image['32x32'](reconstruct_32)

        up2res_64 = self.upscale2['64x64'](reconstruct_32)
        before_select_64 = self.before_select['64x64']\
            (torch.cat([feat_64, mc(resize(pose_face, 64)), initial_64], dim=1))
        reconstruct_64 = self.reconstruct['64x64']\
            (torch.cat([up2res_64, before_select_64, resize(image_32, 64)], dim=1))
        image_64 = self.to_image['64x64'](reconstruct_64)

        up2res_128 = self.upscale2['128x128'](reconstruct_64)
        before_select_128 = self.before_select['128x128']\
            (torch.cat([feat_128, mc(pose_face), initial_128], dim=1))
        reconstruct_128 = self.reconstruct['128x128']\
            (torch.cat([up2res_128, before_select_128, part_combine, check_part_combine,
                        resize(image_64, 128)], dim=1))
        image_128 = self.to_image['128x128'](reconstruct_128)

        return image_128, image_64, image_32


class partRotator(torch.nn.Module):
    """Rotates the facial landmark located patch textures.

       HW: 40x40, 32x40, 32x48
    """
    def __init__(self, num_channels=3, fmap_base=64):
        super(partRotator, self).__init__()

        self.down1 = torch.nn.Sequential(
            _deConvBnActLayer(num_channels, fmap_base, 3, 1, 1, bn=False),
            ResidualBlock(fmap_base),
        )
        self.down2 = torch.nn.Sequential(
            _deConvBnActLayer(fmap_base, 2 * fmap_base, 4, 2, 1),
            ResidualBlock(2 * fmap_base),
        )
        self.down3 = torch.nn.Sequential(
            _deConvBnActLayer(2 * fmap_base, 4 * fmap_base, 4, 2, 1),
            ResidualBlock(4 * fmap_base),
        )

        self.bottleneck = torch.nn.Sequential(
            _deConvBnActLayer(4 * fmap_base, 8 * fmap_base, 4, 2, 1),
            ResidualBlock(8 * fmap_base),
            ResidualBlock(8 * fmap_base),
            _deConvBnActLayer(8 * fmap_base, 4 * fmap_base, 4, 2, 1, transposed=True),
        )

        self.up1 = torch.nn.Sequential(
            _deConvBnActLayer((4 + 4) * fmap_base, 4 * fmap_base, 3, 1, 1),
            ResidualBlock(4 * fmap_base),
            _deConvBnActLayer(4 * fmap_base, 2 * fmap_base, 4, 2, 1, transposed=True),
        )
        self.up2 = torch.nn.Sequential(
            _deConvBnActLayer((2 + 2) * fmap_base, 2 * fmap_base, 3, 1, 1),
            ResidualBlock(2 * fmap_base),
            _deConvBnActLayer(2 * fmap_base, fmap_base, 4, 2, 1, transposed=True),
        )
        self.up3 = torch.nn.Sequential(
            _deConvBnActLayer((1 + 1) * fmap_base, fmap_base, 3, 1, 1),
            ResidualBlock(fmap_base),
        )

        self.check_part = _deConvBnActLayer(fmap_base, num_channels, bn=False, activate='tanh')

        self.apply(_init_weight)

    def forward(self, input):
        c0r = self.down1(input)
        c1r = self.down2(c0r)
        c2r = self.down3(c1r)

        _d1 = self.bottleneck(c2r)

        _d2 = self.up1(torch.cat([_d1, c2r], dim=1))
        _d3 = self.up2(torch.cat([_d2, c1r], dim=1))
        d3r = self.up3(torch.cat([_d3, c0r], dim=1))

        check_part = self.check_part(d3r)

        return d3r, check_part
