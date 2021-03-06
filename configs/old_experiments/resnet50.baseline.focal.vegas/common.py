'''*************************************************************************
	> File Name: common.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 01:46:28 PM EDT
 ************************************************************************'''
import json
with open('../../scripts/mean_std.json') as f:
    mean_std = json.load(f)

class Config:
    #Model
    backbone = 'resnet50' #backbone name ['resnet50', 'xception', 'drn', 'mobilenet', 'resnet101']
    out_stride = 16 #network output stride

    #Data
    all_dataset = ['Shanghai', 'Vegas', 'Paris', 'Khartoum']
    dataset = 'Vegas'
    source_dist = mean_std[dataset]
    target = 'Shanghai'
    target_dist = mean_std[target]
    train_num_workers = 4
    val_num_workers = 2
    img_root = '/usr/xtmp/satellite/spacenet/'
    #Train
    batch_size = 16
    freeze_bn = False
    sync_bn = False
    loss = 'focal' #['ce', 'focal']
    epochs = 1000
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    lr_scheduler = 'cos'
    lr_step = 5
    warmup_epochs = 50
    lambda_adv = 0.001



config = Config()
