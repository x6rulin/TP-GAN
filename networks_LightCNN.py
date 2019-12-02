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
            torch.nn.Sequential(
                torch.nn.MaxPool2d(2, 2),
                _DFBlock(48, 48, 96, 1),),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(2, 2),
                _DFBlock(96, 96, 192, 2),),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(2, 2),
                _DFBlock(192, 128, 128, 3),
                _DFBlock(128, 128, 128, 4)),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Sequential(
                torch.nn.Linear(128 * (resolution >> 4) ** 2, 512),
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
