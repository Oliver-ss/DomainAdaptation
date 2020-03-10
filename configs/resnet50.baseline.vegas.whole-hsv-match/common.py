'''*************************************************************************
	> File Name: common.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 01:46:28 PM EDT
 ************************************************************************'''
import json
with open('../../scripts/mean_std.json') as f:
    mean_std = json.load(f)

with open('../../scripts/building_hsv_hist.json') as f:
    building_hsv_hist = json.load(f)

with open('../../scripts/nonbuilding_hsv_hist.json') as f:
    nonbuilding_hsv_hist = json.load(f)

with open('../../scripts/building_rgb_hist.json') as f:
    building_rgb_hist = json.load(f)

with open('../../scripts/nonbuilding_rgb_hist.json') as f:
    nonbuilding_rgb_hist = json.load(f)

with open('../../scripts/rgb_hist.json') as f:
    rgb_hist = json.load(f)

with open('../../scripts/hsv_hist.json') as f:
    hsv_hist = json.load(f)

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
    channels = ['H', 'S', 'V']

    if 'red' in channels:
        building_color_dist = building_rgb_hist
        nonbuilding_color_dist = nonbuilding_rgb_hist
        color_dist = rgb_hist
    else:
        building_color_dist = building_hsv_hist
        nonbuilding_color_dist = nonbuilding_hsv_hist
        color_dist = hsv_hist

    train_num_workers = 4
    val_num_workers = 2
    img_root = '/usr/xtmp/satellite/spacenet/'
    #Train
    batch_size = 8
    freeze_bn = False
    sync_bn = False
    loss = 'ce' #['ce', 'focal']
    epochs = 300
    lr = 1e-3
    lr_ratio = 10
    momentum = 0.9
    weight_decay = 5e-4
    lr_scheduler = 'poly'
    lr_step = 5
    warmup_epochs = 50
    lambda_adv = 0.001



config = Config()
