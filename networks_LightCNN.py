r"""Light CNN framework for face analysis and recognition.

 .. _LightCNN: using Max-Feature-Map to simulate neural inhibition.
     https://arxiv.org/pdf/1511.02683.pdf
 """
import pickle
import torch


class DeepFace(torch.nn.Module):
    """LightCNN, mapping faces to the compact deep feature space. """
    def __init__(self, num_channels=1, resolution=128, pre_trained=True):
        super(DeepFace, self).__init__()

        self.feature_extract = torch.nn.ModuleList([
            _Conv2dMFM2_1(num_channels, 48, 5, 1, 2),
            torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, ceil_mode=True),
                                _DFBlock(48, 48, 96, 1),),
            torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, ceil_mode=True),
                                _DFBlock(96, 96, 192, 2),),
            torch.nn.Sequential(torch.nn.MaxPool2d(2, 2, ceil_mode=True),
                                _DFBlock(192, 128, 128, 3),
                                _DFBlock(128, 128, 128, 4),),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.Sequential(torch.nn.Linear(128 * (resolution >> 4) ** 2, 2 * 256),
                                MFM2_1(),),
        ])

        if pre_trained:
            self.load_state_dict(self.__loadpickle())

    def forward(self, input):
        features = []
        for ext in self.feature_extract[:-1]:
            input = ext(input)
            features.append(input)
        features.append(self.feature_extract[-1](input.reshape(input.shape[0], -1)))

        return features

    def __loadpickle(self, path="DeepFace168.pickle"):
        with open(path, 'rb') as _pf:
            data_dict = pickle.load(_pf, encoding='iso-8859-1')
            keys = sorted(data_dict.keys())

            tmp = []
            for i in range(1, len(keys) - 4):
                if len(keys[i]) == 5:
                    tmp.append(keys.pop(i))

            idx = [4, 10, 18, 28]
            for i, k in enumerate(tmp):
                keys.insert(idx[i], k)

        params = []
        for _k in keys[:-1]:
            params.extend([torch.from_numpy(data_dict[_k][0]).permute(3, 2, 0, 1),
                           torch.from_numpy(data_dict[_k][1])])
        params.extend([torch.from_numpy(data_dict[keys[-1]][0]).permute(1, 0),
                       torch.from_numpy(data_dict[keys[-1]][1])])

        state_dict = self.state_dict()
        for i, k in enumerate(state_dict.keys()):
            state_dict[k] = params[i]

        return state_dict


class _DFBlock(torch.nn.Module):

    def __init__(self, in_channels, mix_channels, out_channels, res_num):
        super(_DFBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            *[ResidualBlock(in_channels, 3, 1, 1) for _ in range(res_num)],
            _Conv2dMFM2_1(in_channels, mix_channels, 1, 1, 0),
            _Conv2dMFM2_1(mix_channels, out_channels, 3, 1, 1),
        )

    def forward(self, input):
        return self.sub_module(input)


class ResidualBlock(torch.nn.Module):
    """Residual block. """
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _Conv2dMFM2_1(channels, channels, kernel_size, stride, padding),
            _Conv2dMFM2_1(channels, channels, kernel_size, stride, padding),
        )

    def forward(self, input):
        return input + self.sub_module(input)


class _Conv2dMFM2_1(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(_Conv2dMFM2_1, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 2 * out_channels, kernel_size, stride, padding, bias=bias),
            MFM2_1(),
        )

    def forward(self, input):
        return self.sub_module(input)


class MFM2_1(torch.nn.Module):
    """Max-Feature-Map (MFM) 2/1 operation. """
    def forward(self, input):
        input = input.reshape((input.shape[0], 2, -1, *input.shape[2:]))
        output = input.max(dim=1)[0]

        return output


class MFM3_2(torch.nn.Module):
    """Max-Feature-Map (MFM) 3/2 operation. """
    def forward(self, input):
        input = input.reshape(input.shape[0], 3, -1, *input.shape[2:])
        output = torch.cat([input.max(dim=1)[0], input.median(dim=1)[0]], dim=1)

        return output


class LightCNN_4(torch.nn.Module):
    """Constructed by 4 convolution layers with Max-Feature-Map operations
       and 4 max-pooling layers, like Alexnet.

       Input size: 128x128
    """
    def __init__(self, num_channels=1):
        super(LightCNN_4, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            _Conv2dMFM2_1(num_channels, 48, 9, 1, 0),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Conv2dMFM2_1(48, 96, 5, 1, 0),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Conv2dMFM2_1(96, 128, 5, 1, 0),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Conv2dMFM2_1(128, 192, 4, 1, 0),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        self.mfm_fc = torch.nn.Sequential(
            torch.nn.Linear(192 * 5 * 5, 2 * 256),
            MFM2_1(),
        )

    def forward(self, input):
        output = self.feature_extract(input)
        output = self.mfm_fc(output.reshape(output.size(0), -1))

        return output


class LightCNN_9(torch.nn.Module):
    """Integrates NIN and a small convolution kernel size in to the network with MFM. """
    def __init__(self, num_channels=1, resolution=128):
        super(LightCNN_9, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            _Conv2dMFM2_1(num_channels, 48, 5, 1, 2),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Group(48, 96, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Group(96, 192, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            _Group(192, 128, 3, 1, 1),
            _Group(128, 128, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        self.mfm_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * (resolution >> 4) ** 2, 2 * 256),
            MFM2_1(),
        )

    def forward(self, input):
        output = self.feature_extract(input)
        output = self.mfm_fc(output.reshape(output.size(0), -1))

        return output


class LightCNN_29(torch.nn.Module):
    """Introduces the idea of residual block to LightCNN. """
    def __init__(self, num_channels=1, resolution=128):
        super(LightCNN_29, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            _Conv2dMFM2_1(num_channels, 48, 5, 1, 2),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(48) for _ in range(1)],
            _Group(48, 96, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(96) for _ in range(2)],
            _Group(96, 192, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(192) for _ in range(3)],
            _Group(192, 128, 3, 1, 1),
            *[ResidualBlock(128) for _ in range(4)],
            _Group(128, 128, 3, 1, 1),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
        )
        self.mfm_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * (resolution >> 4) ** 2, 2 * 256),
            MFM2_1(),
        )

    def forward(self, input):
        output = self.feature_extract(input)
        output = self.mfm_fc(output.reshape(output.size(0), -1))

        return output


class LightCNN_29v2(torch.nn.Module):
    """Replaces the MaxPool2d with MaxPool2d + AvgPool2d to preserve more
       semantic and spatial information.
    """
    def __init__(self, num_channels=1, resolution=128):
        super(LightCNN_29v2, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            _Conv2dMFM2_1(num_channels, 48, 5, 1, 2),
            _MixPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(48) for _ in range(1)],
            _Group(48, 96, 3, 1, 1),
            _MixPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(96) for _ in range(2)],
            _Group(96, 192, 3, 1, 1),
            _MixPool2d(2, 2, ceil_mode=True),
            *[ResidualBlock(192) for _ in range(3)],
            _Group(192, 128, 3, 1, 1),
            *[ResidualBlock(128) for _ in range(4)],
            _Group(128, 128, 3, 1, 1),
            _MixPool2d(2, 2, ceil_mode=True),
        )
        self.mfm_fc = torch.nn.Sequential(
            torch.nn.Linear(128 * (resolution >> 4) ** 2, 2 * 256),
            MFM2_1(),
        )

    def forward(self, input):
        output = self.feature_extract(input)
        output = self.mfm_fc(output.reshape(output.size(0), -1))

        return output


class _Group(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(_Group, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _Conv2dMFM2_1(in_channels, in_channels, 1, 1, 0),
            _Conv2dMFM2_1(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, input):
        return self.sub_module(input)


class _MixPool2d(torch.nn.Module):

    def __init__(self, kernel_size, stride, padding=0, ceil_mode=False):
        super(_MixPool2d, self).__init__()

        self.max_pool = torch.nn.MaxPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size, stride, padding, ceil_mode=ceil_mode)

    def forward(self, input):
        return self.max_pool(input) + self.avg_pool(input)
