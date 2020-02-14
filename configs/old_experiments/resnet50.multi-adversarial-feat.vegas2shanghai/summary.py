import sys, os
sys.path.append(os.getcwd())
from model.sync_batchnorm.replicate import patch_replication_callback
from model.deeplab import *
from model.discriminator import Discriminator
import torch
from torchsummary import summary
from common import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepLab(num_classes=2,
                backbone=config.backbone,
                output_stride=config.out_stride,
                sync_bn=config.sync_bn,
                freeze_bn=config.freeze_bn).to(device)
backbone = model.backbone
aspp = model.aspp
decoder = model.decoder

D = Discriminator(num_classes=256, ndf=16)

summary(D, (256, 50, 50))
