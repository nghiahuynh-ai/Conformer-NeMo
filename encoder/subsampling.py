import math
import torch


class Subsampling(torch.nn.Module):

    def __init__(self, subsampling, subsampling_factor, feat_in, feat_out, conv_channels, activation=torch.nn.ReLU()):
        super(Subsampling, self).__init__()

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        in_channels = 1
        layers = []
        self._ceil_mode = False

        if subsampling == 'vggnet':
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels
        elif subsampling == 'striding':
            self._padding = 1
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels
        elif subsampling == 'resnet':
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True
            
            for i in range(self._sampling_num):
                block = ResNetBlock(in_features=in_channels, out_features=conv_channels)
                layers.append(block)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths


class ResNetBlock(torch.nnModule):
    def __init__(self, in_features, out_features, activation=torch.nn.ReLU()):
        super(ResNetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=out_features)
        self.conv2 = torch.nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=out_features)
        self.activation = activation
        
    def forward(self, x):
        x = x + self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = x + self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        return x
        

def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
