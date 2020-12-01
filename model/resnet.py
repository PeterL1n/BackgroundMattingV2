from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck


class ResNetEncoder(ResNet):
    """
    ResNetEncoder inherits from torchvision's official ResNet. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    global average pooling layer and the fully connected layer that was originally
    used for classification. The forward method  additionally returns the feature
    maps at all resolutions for decoder's use.
    """
    
    layers = {
        'resnet50':  [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }
    
    def __init__(self, in_channels, variant='resnet101', norm_layer=None):
        super().__init__(
            block=Bottleneck,
            layers=self.layers[variant],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=norm_layer)
        
        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            
        # Delete fully-connected layer
        del self.avgpool
        del self.fc
    
    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0
