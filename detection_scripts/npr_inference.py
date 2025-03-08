import sys
import time
import os
import csv
import torch
import numpy as np
import random
from collections import OrderedDict
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

DetectionTests = {
    'SD21': {
        'dataroot': '/path/to/your/dataset/twitter/SD21/test/',
        'no_resize': False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
        'no_crop': True,
    },
    # Uncomment and configure other datasets as needed
    # 'GANGen-Detection': {
    #     'dataroot': '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
    #     'no_resize': True,
    #     'no_crop': True,
    # },
    # 'DiffusionForensics': {
    #     'dataroot': '/opt/data/private/DeepfakeDetection/DiffusionForensics/',
    #     'no_resize': False,
    #     'no_crop': True,
    # },
    # 'UniversalFakeDetect': {
    #     'dataroot': '/opt/data/private/DeepfakeDetection/UniversalFakeDetect/',
    #     'no_resize': False,
    #     'no_crop': True,
    # },
}

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# Load the checkpoint
checkpoint = torch.load(opt.model_path, map_location='cpu')

# Extract the model state dict
model_state_dict = checkpoint['model']

# Remove the 'module.' prefix from the keys if present
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    if k.startswith('module.'):
        new_key = k[7:]  # remove 'module.' prefix
    else:
        new_key = k
    new_state_dict[new_key] = v

# Initialize your model
model = resnet50(num_classes=1)

# Load the state dict into your model
model.load_state_dict(new_state_dict, strict=True)

# Move model to GPU
model.cuda()
model.eval()

for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    printSet(testSet)

    accs = []
    aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' # os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
        
        accs.append(acc)
        print('accs',accs)
        aps.append(ap)
        print('AUC: {:2.2f}, AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(auc*100., ap*100., acc*100., r_acc*100., f_acc*100.))
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1, 'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
    print('*'*25)

