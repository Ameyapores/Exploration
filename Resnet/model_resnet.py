import torch.nn as nn
import torch
import torch.nn.functional as F
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.elu(self.conv1(x))

        out = F.elu(self.conv2(out))
        #print (out.size())
        if self.downsample is not None:
            residual = F.elu(self.downsample(x))
            #print (residual.size())

        out = out + residual
        out = F.elu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, num_inputs, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(32)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride)
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        
        #x = F.elu(self.conv2(x))

        #x = F.elu(self.conv2(x))

        #x = F.elu(self.conv2(x))
        
        #x = self.maxpool(x)
        #print (x.size())
        x = self.layer1(x)
        #x = self.layer1(x)
        #print (x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        #print (x.size())
        x = self.layer4(x)

        #x = self.avgpool(x)
        #print (x.size())
        #x = x.view(x.size(0), -1)
        #print (x.size())
        #x = self.fc(x)

        return x

def resnet_new(num_inputs, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_inputs, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

'''model = resnet18(num_inputs= 1)
x = torch.randn(1, 84, 84)
x.unsqueeze_(0)
print (x.shape)
out = model(x)
print (out.shape)'''