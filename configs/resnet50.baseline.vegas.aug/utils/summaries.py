'''*************************************************************************
	> File Name: summaries.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Thu 05 Dec 2019 10:26:01 PM EST
 ************************************************************************'''
import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import json
import numpy as np
import cv2

with open('../../scripts/mean_std.json') as f:
    dist = json.load(f)

def recover_images(city, images):
    mean, std = dist[city]['mean'], dist[city]['std']
    for im in images:
        for i, c in enumerate(im):
            c *= std[i]
            c += mean[i]
    return images

def color_images(pred, target):
    imgs = []
    for p, t in zip(pred, target):
        tmp = p * 2 + t
        np.squeeze(tmp)
        img = np.zeros((p.shape[0], p.shape[1], 3))
        # bkg:negative, building:postive
        img[np.where(tmp==0)] = [0, 0, 0] # Black RGB, for true negative
        img[np.where(tmp==1)] = [255, 0, 0] # Red RGB, for false negative
        img[np.where(tmp==2)] = [0, 255, 0] # Green RGB, for false positive
        img[np.where(tmp==3)] = [255, 255, 0] #Yellow RGB, for true positive
        imgs.append(img.transpose(2,0,1))
    imgs = np.array(imgs)
    return imgs

def fuse_images(imgs, labels):
    ans = []
    for im, label in zip(imgs, labels):
        im = im.transpose(1,2,0).astype(np.float32)
        mask = np.zeros_like(im)
        mask[:,:,0] = label
        mask.astype(np.float32)
        img = cv2.addWeighted(im, 0.75, mask, 0.25, 0)
        img = np.array(img)
        img = img.transpose(2,0,1)
        ans.append(img)
    return np.array(ans)

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))

    def visualize_image(self, prefix, city, image, target, output, global_step, size=10):
        images = image[:size].clone().cpu().data
        images = recover_images(city, images)
        grid_image = make_grid(images, 5, normalize=False)
        self.writer.add_image(prefix+'/Image', grid_image, global_step)
        pred = np.argmax(output[:size].detach().cpu().numpy(), 1)
        target = target[:size].detach().cpu().numpy()
        overlap_pred = fuse_images(images.cpu().numpy(), pred)
        overlap_gt = fuse_images(images.cpu().numpy(), target)
        grid_image = make_grid(torch.from_numpy(overlap_pred), 5, normalize=False)
        self.writer.add_image(prefix+'/Image-Pred', grid_image, global_step)
        grid_image = make_grid(torch.from_numpy(overlap_gt), 5, normalize=False)
        self.writer.add_image(prefix+'/Image-GT', grid_image, global_step)

        #grid_image = make_grid(torch.from_numpy(pred[:,None,:,:]), 5, normalize=False)
        #self.writer.add_image(prefix+'/Prediction', grid_image, global_step)
        #grid_image = make_grid(torch.from_numpy(target[:,None,:,:]), 5, normalize=False)
        #self.writer.add_image(prefix+'/GT', grid_image, global_step)
        grid_image = make_grid(torch.from_numpy(color_images(pred, target)), 5, normalize=True, range=(0,255))
        self.writer.add_image(prefix+'/Color', grid_image, global_step)

        #grid_image = make_grid(images, 5, normalize=False)
        #self.writer.add_image(prefix+'/Image', grid_image, global_step)
if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from data import spacenet
    from torch.utils.data import DataLoader
    summary = TensorboardSummary('log')

    dataset = spacenet.Spacenet('Vegas', source_dist = dist['Vegas'])
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    for i, sample in enumerate(loader):
        image, target = sample['image'], sample['label']
        summary.visualize_image('train', 'Vegas', image, target, image[:,:2,:,:], i)
        if i == 3:
            break

