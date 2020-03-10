'''*************************************************************************
	> File Name: spacenet.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Mon 21 Oct 2019 04:01:05 PM EDT
 ************************************************************************'''
import os
import numpy as np
import scipy.misc as m
import cv2
from torch.utils import data
from torchvision import transforms
import data.custom_transforms as tr
#import custom_transforms as tr
import json
from PIL import Image

def GetTable(pdf_s, pdf_t):
    cdf_s = pdf_s.cumsum() / pdf_s.sum()
    cdf_t = pdf_t.cumsum() / pdf_t.sum()
    return np.interp(cdf_s, cdf_t, np.arange(256))

def GetAllTable(hist, source, target, channels):
    table = {}
    if hist != {}:
        for c in channels:
            table[c] = GetTable(np.array(hist[source][c]), np.array(hist[target][c]))
    return table

class Spacenet(data.Dataset):
    NUM_CLASSES = 2
    def __init__(self, city='Vegas', split='train', img_root='/usr/xtmp/satellite/spacenet/',
                 source_dist={'mean':(0.,0.,0.),'std':(1.,1.,1.,)}, if_pair=False, target='Shanghai', random_match=False):
        self.img_root = img_root
        #self.name_root = '../../dataset/spacenet/domains/' + city
        self.name_root = '../../dataset/spacenet/domains/' + city
        with open(os.path.join(self.name_root, split + '.json')) as f:
            self.files = json.load(f)
        self.source_dist = source_dist
        self.split = split
        self.classes = [0, 1]
        self.class_names = ['bkg', 'building']
        self.if_pair = if_pair
        self.random_match = random_match
        self.target = target
        self.target_name_root = '../../dataset/spacenet/domains/' + target
        with open(os.path.join(self.target_name_root, split + '.json')) as f:
            self.target_files = json.load(f)

        if not self.files:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files), split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #img = cv2.imread(os.path.join(self.img_root, self.files[index] + '_RGB.tif'))
        img = Image.open(os.path.join(self.img_root, self.files[index] + '_RGB.tif')).convert('RGB')
        label = Image.open(os.path.join(self.img_root, self.files[index] + '_GT.tif'))
        reference = Image.open(os.path.join(self.img_root, self.target_files[index if index < len(self.target_files) else index % len(self.target_files)] + '_RGB.tif')).convert('RGB')
        sample = {'image': img, 'label': label, 'reference': reference}
        if self.split == 'train':
            if self.if_pair:
                return self.transform_pair_train(sample)
            else:
                return self.transform_tr(sample)
        elif self.split == 'val':
            if self.if_pair:
                return self.transform_pair_val(sample)
            else:
                return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def transform_tr(self, sample):
        if not self.random_match:
            composed_transforms = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=400, crop_size=400, fill=0),
                #tr.Remap(self.building_table, self.nonbuilding_table, self.channels)
                tr.RandomGaussianBlur(),
                #tr.ConvertFromInts(),
                #tr.PhotometricDistort(),
                tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std']),
                tr.ToTensor(),
            ])
        else:
            composed_transforms = transforms.Compose([
                tr.HistogramMatching(),
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(base_size=400, crop_size=400, fill=0),
                tr.RandomGaussianBlur(),
                #tr.ConvertFromInts(),
                #tr.PhotometricDistort(),
                tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std']),
                tr.ToTensor(),
            ])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(400),
            tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std']),
            tr.ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_pair_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(400),
            tr.HorizontalFlip(),
            tr.GaussianBlur(),
            tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std'], if_pair=True),
            tr.ToTensor(if_pair=True),
        ])
        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=400),
            tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std']),
            tr.ToTensor(),
        ])
        return composed_transforms(sample)

    def transform_pair_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=400, crop_size=400, fill=0),
            tr.HorizontalFlip(),
            tr.GaussianBlur(),
            tr.Normalize(mean=self.source_dist['mean'], std=self.source_dist['std'], if_pair=True),
            tr.ToTensor(if_pair=True),
        ])
        return composed_transforms(sample)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    def recover_image(image, dist):
        mean, std = dist['mean'], dist['std']
        for i, c in enumerate(image):
            c *= std[i]
            c += mean[i]
        return image
    with open('../../../scripts/mean_std.json') as f:
        mean_std = json.load(f)
    with open('../../../scripts/color.json') as f:
        color = json.load(f)
    source_dist = mean_std['Vegas']
    color_dist = color['Shanghai']
    spacenet_train = Spacenet(city='Vegas', split='train', img_root='/data/spacenet/',
                    source_dist=source_dist, if_pair=False, color_dist=color_dist)
    dataloader = DataLoader(spacenet_train, batch_size=1, shuffle=False, num_workers=1)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'][jj].numpy()
            gt = sample['label'][jj].numpy()
            gt = gt[:,:,None]
            gt_ = gt.repeat(3, axis=2)
            img = recover_image(img, source_dist)
            img = img.transpose(1,2,0)
            print(gt.max())
            #print(img.shape)
            #cv2.imshow('img', img[:,:,::-1])
            show = np.hstack((img, gt_))
            cv2.imshow('show', show[:,:,::-1])
            c = chr(cv2.waitKey(0) & 0xff)
            if c == 'q':
                exit()





