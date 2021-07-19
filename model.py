from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
from src import resnet_models as resnet_models

def Detector(MODEL_SELECT, NUM_SPOOF_CLASS, GATE_REDUCTION=4):

    if MODEL_SELECT == 0:
        print('using ResNet34.')
        model = resnet_models.resnet34(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 1:
        print('using SEResNet34.')
        model = resnet_models.se_resnet34(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 2:
        print('using ResNet50.')
        model = resnet_models.resnet50(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 3:
        print('using SEResNet50.')
        model = resnet_models.se_resnet50(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 4:
        print('using Res2Net50_26w_4s.')
        model = resnet_models.res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 5:
        print('using SERes2Net50_26w_4s.')
        model = resnet_models.se_res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True)

    elif MODEL_SELECT == 6:
        print('using SCG-Res2Net50_26w_4s.')
        model = resnet_models.se_gated_linear_res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True, gate_reduction=GATE_REDUCTION)

    elif MODEL_SELECT == 7:
        print('using MCG-Res2Net50_26w_4s.')
        model = resnet_models.se_gated_linearconcat_res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True, gate_reduction=GATE_REDUCTION)

    elif MODEL_SELECT == 8:
        print('using MLCG-Res2Net50_26w_4s.')
        model = resnet_models.se_gated_nonlinearconcat_res2net50_v1b(num_classes=NUM_SPOOF_CLASS, KaimingInit=True, gate_reduction=GATE_REDUCTION)

    return model 

def test_Detector(model_id=6):
    model_params = {
        "MODEL_SELECT" : model_id,
        "NUM_SPOOF_CLASS" : 2,
        "GATE_REDUCTION" : 4,
    }
    print('model_id', model_id)
    model = Detector(**model_params)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model contains {} parameters'.format(model_params))
    # print(model)
    x = torch.randn(2,1,257,400)
    output = model(x)
    print(x.size())
    # print(output.size())
    print(output)

if __name__ == '__main__':
    for id in range(0, 9):
        test_Detector(id)

