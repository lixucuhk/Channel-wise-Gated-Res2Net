import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from src.resnet_blocks import SELayer, BasicBlock, SEBasicBlock, Bottleneck, SEBottleneck, Bottle2neck, SEBottle2neck, SEGatedLinearBottle2neck, SEGatedLinearConcatBottle2neck, SEGatedNonlinearConcatBottle2neck
from src.pooling import StatsPooling
from src.oc_softmax import OCAngleLayer, OCSoftmaxWithLoss
from src.a_softmax import AngleLayer, AngularSoftmaxWithLoss
from src.am_softmax import AMAngleLayer, AMSoftmaxWithLoss

class ResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes, KaimingInit=False, loss='softmax'):
        
        self.inplanes = 16

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.loss = loss

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Sequential(nn.Linear(128*block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()

        elif self.loss == 'a-softmax':
            self.cls_layer = AngleLayer(128*block.expansion, num_classes, m=4)
            self.loss_F = AngularSoftmaxWithLoss()

        elif self.loss == 'am-softmax':
            self.cls_layer = AMAngleLayer(128*block.expansion, num_classes, s=20, m=0.9)
            self.loss_F = AMSoftmaxWithLoss()

        elif self.loss == 'oc-softmax':
            self.cls_layer = OCAngleLayer(128*block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        if KaimingInit == True:
            print('Using Kaiming Initialization.')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())

        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())

        x = self.avgpool(x).view(x.size()[0], -1)

        #print(x.shape)
        x = self.cls_layer(x)
        #print(out.shape)
        return x
        # return F.log_softmax(out, dim=-1)

class GatedRes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000, loss='softmax', gate_reduction=4, **kwargs):
        self.inplanes = 16
        super(GatedRes2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.gate_reduction = gate_reduction
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])#64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)#512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Sequential(nn.Linear(128*block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()

        elif self.loss == 'a-softmax':
            self.cls_layer = AngleLayer(128*block.expansion, num_classes, m=4)
            self.loss_F = AngularSoftmaxWithLoss()

        elif self.loss == 'am-softmax':
            self.cls_layer = AMAngleLayer(128*block.expansion, num_classes, s=20, m=0.9)
            self.loss_F = AMSoftmaxWithLoss()

        elif self.loss == 'oc-softmax':
            self.cls_layer = OCAngleLayer(128*block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale,
                  gate_reduction=self.gate_reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale,
                      gate_reduction=self.gate_reduction))

        return nn.Sequential(*layers)

    def _forward(self, x):
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)

        return x
        # return F.log_softmax(x, dim=-1)

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000, loss='softmax', **kwargs):
        self.inplanes = 16
        super(Res2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])#64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)#512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Sequential(nn.Linear(128*block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()

        elif self.loss == 'a-softmax':
            self.cls_layer = AngleLayer(128*block.expansion, num_classes, m=4)
            self.loss_F = AngularSoftmaxWithLoss()

        elif self.loss == 'am-softmax':
            self.cls_layer = AMAngleLayer(128*block.expansion, num_classes, s=20, m=0.9)
            self.loss_F = AMSoftmaxWithLoss()

        elif self.loss == 'oc-softmax':
            self.cls_layer = OCAngleLayer(128*block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale))

        return nn.Sequential(*layers)

    def _forward(self, x):
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)

        return x

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward

''' ResNet models'''
def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet18(**kwargs):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet50(**kwargs):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model

'''Res2Net models'''
def res2net50_v1b(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)

    return model

def se_res2net50_v1b(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model

def res2net50_v1b_14w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    return model

def se_res2net50_v1b_14w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    return model

def res2net50_v1b_26w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    return model

def se_res2net50_v1b_26w_8s(**kwargs):
    """Constructs a Res2Net-50_v1b model.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    """
    model = Res2Net(SEBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    return model

'''GatedRes2Net models'''

def se_gated_linear_res2net50_v1b(**kwargs):
    """Constructs a SE-Gated-Res2Net-50_v1b model.
    SE-Gated-Res2Net-50 refers to the SE-Gated-Res2Net-50_v1b_26w_4s.
    """
    model = GatedRes2Net(SEGatedLinearBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model

def se_gated_linearconcat_res2net50_v1b(**kwargs):
    model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model

def se_gated_nonlinearconcat_res2net50_v1b(**kwargs):
    model = GatedRes2Net(SEGatedNonlinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.rand(2, 1, 257, 400).to(device)
    label = torch.randint(0, 2, (2,)).long().to(device)
    model = se_gated_linearconcat_res2net50_v1b(pretrained=False, num_classes=3, loss='oc-softmax')
    model.to(device)
    output = model(images)
    print(images.size())
    print(output)

