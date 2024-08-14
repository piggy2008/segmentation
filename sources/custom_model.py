import torchvision
from torchvision import models
import torch
from pidnet import PIDNet

class DeepLabV3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)['out']
        return output

def initialize_model(num_classes, keep_feature_extract=False, backbone='resnet101', use_pretrained=True):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    if backbone == 'resnet101':
        model_deeplabv3 = models.segmentation.deeplabv3_resnet101(pretrained=use_pretrained, progress=True)
    else:
        model_deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=use_pretrained, progress=True)


    model_deeplabv3.aux_classifier = None
    if keep_feature_extract:
        for param in model_deeplabv3.parameters():
            param.requires_grad = False

    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model_deeplabv3

def initialize_model_pidnet(num_classes, backbone='pidnet_s', use_pretrained=True, model_path=None):
    """ DeepLabV3 pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    """
    if 's' in backbone:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in backbone:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True)
    if model_path is not None:
        if use_pretrained:
            pretrained_state = torch.load(model_path, map_location='cpu')['state_dict']
            model_dict = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(pretrained_state)
            msg = 'Loaded {} parameters!'.format(len(model_dict))
            print('Attention!!!')
            print(msg)
            print('Over!!!')
            model.load_state_dict(model_dict, strict = False)
        else:
            pretrained_dict = torch.load(model_path, map_location='cpu')
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
            msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
            print('Attention!!!')
            print(msg)
            print('Over!!!')

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict = False)

    return model
