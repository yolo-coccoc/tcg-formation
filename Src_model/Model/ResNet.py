import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inp_channels: int = 64, out_channels: int = 64, downsample: bool = False):
        super().__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            inp_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if downsample or inp_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity
        return self.relu(x)


class ResNet18_Classification(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        num_residual_blocks: list[int] = [2, 2, 2, 2],
        num_classes: int = 1,
    ):
        super().__init__()

        self.initial_channels = 64
        self.conv1 = nn.Conv2d(
            inp_channels,
            self.initial_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet, final_channels = self._build_resnet(num_residual_blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)

    def _build_resnet(self, num_residual_blocks: list[int]) -> tuple[nn.Sequential, int]:
        layers: list[nn.Module] = []
        inp_channels = self.initial_channels

        for stage_index, block_count in enumerate(num_residual_blocks):
            out_channels = self.initial_channels * (2**stage_index)
            for block_index in range(block_count):
                downsample = stage_index > 0 and block_index == 0
                layers.append(ResidualBlock(inp_channels, out_channels, downsample))
                inp_channels = out_channels

        return nn.Sequential(*layers), inp_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


if __name__ == '__main__':
    model = ResNet18_Classification(inp_channels = 228, 
                                    num_residual_blocks = [2, 2, 2, 2], 
                                    num_classes = 1)#.to('cuda')
    
    # print(summary(model, 
    #               [(228, 33, 33)]))
    
    sample = torch.rand(64, 228, 33, 33)#.to('cuda')
    print(model(sample).shape)