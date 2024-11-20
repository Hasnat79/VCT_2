import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import argparse
import os
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model



#from utils.utils import get_network, str2bool, to_cuda

def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on" "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True
        
def to_cuda(data, device="cuda", exclude_keys: "list[str]" = None):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, (tuple, list, set)):
        data = [to_cuda(b, device) for b in data]
    elif isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        for k in data.keys():
            if k not in exclude_keys:
                data[k] = to_cuda(data[k], device)
    else:
        # raise TypeError(f"Unsupported type: {type(data)}")
        data = data
    return data

import torch.nn as nn
from importlib import import_module

def get_network(arch: str, isTrain=False, continue_train=False, init_gain=0.02, pretrained=True):
    if "resnet" in arch:
        if arch == "resnet50":
            from torchvision.models import resnet50 as resnet
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

        if isTrain:
            if continue_train:
                model = resnet(num_classes=1)
            else:
                model = resnet(pretrained=pretrained)
                model.fc = nn.Linear(2048, 1)
                nn.init.normal_(model.fc.weight.data, 0.0, init_gain)
        else:
            model = resnet(num_classes=1)
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-f", "--file", default="data/test/lsun_adm/1_fake/0.png", type=str, help="path to image file or directory of images"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/lsun_adm/model_epoch_latest.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)

args = parser.parse_args()

if os.path.isfile(args.file):
    print(f"Testing on image '{args.file}'")
    file_list = [args.file]
elif os.path.isdir(args.file):
    file_list = sorted(glob.glob(os.path.join(args.file, "*.jpg")) + glob.glob(os.path.join(args.file, "*.png"))+glob.glob(os.path.join(args.file, "*.JPEG")))
    print(f"Testing images from '{args.file}'")
else:
    raise FileNotFoundError(f"Invalid file path: '{args.file}'")


model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()
if not args.use_cpu:
    model.cuda()

print("*" * 50)

trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    )
)
for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
    img = Image.open(img_path).convert("RGB")
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    if not args.use_cpu:
        in_tens = in_tens.cuda()

    with torch.no_grad():
        prob = model(in_tens).sigmoid().item()
    print(f"Prob of being synthetic: {prob:.4f}")
