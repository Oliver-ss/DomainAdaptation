'''*************************************************************************
	> File Name: generate_fake_dataset.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Thu 20 Feb 2020 12:21:40 PM EST
 ************************************************************************'''
import random
import numpy as np
import cv2
import json
import os
from tqdm import tqdm

def TranslateDict(dist):
    new_dist = {}
    for city in dist.keys():
        red = np.cumsum(list(dist[city]['red'].values()))
        green = np.cumsum(list(dist[city]['green'].values()))
        blue = np.cumsum(list(dist[city]['blue'].values()))
        new_dist[city] = {'red':red, 'green':green, 'blue':blue}
    return new_dist

class BuildingRepaint(object):
    def __init__(self, building_dist, nonbuilding_dist, source='Vegas', target='Shanghai'):
        self.building_dist = TranslateDict(building_dist)
        self.nonbuilding_dist = TranslateDict(nonbuilding_dist)

        self.source = source
        self.target = target

    def prob2color(self, prob, city, if_building):
        dist = self.building_dist[city] if if_building else self.nonbuilding_dist[city]
        ans = {}
        for c in ['red', 'green', 'blue']:
            for i, p in enumerate(dist[c]):
                if prob[c] < p:
                    ans[c] = i
                    break
            if c not in ans.keys():
                ans[c] = 255
        return ans

    def color2prob(self, color, city, if_building):
        dist = self.building_dist if if_building else self.nonbuilding_dist
        r, g, b = np.int0(color)
        r_ = dist[city]['red'][r-1] if r>0 else 0
        g_ = dist[city]['green'][g-1] if g>0 else 0
        b_ = dist[city]['blue'][b-1] if b>0 else 0
        return {'red':r_, 'green':g_, 'blue':b_}

    #def choose_color(self):
        #r_ = random.random()
        #g_ = random.random()
        #b_ = random.random()
        #print(r_,g_,b_)
        #r = self.find_prob(r_, self.dist['red'])
        #g = self.find_prob(r_, self.dist['green'])
        #b = self.find_prob(r_, self.dist['blue'])
        #return (r, g, b)

    def GaussianBlur(self, img, kernel=(5,5)):
        return cv2.GaussianBlur(img, kernel, 0)

    def GenerateImage(self, img, mask):
        row, col = mask.shape[:2]
        for i in range(row):
            for j in range(col):
                if mask[i][j][0] == 1.:
                    color = self.prob2color(self.color2prob(img[i][j], self.source, True), self.target, True)
                    img[i][j] = np.array([color['red'], color['green'], color['blue']])
                elif mask[i][j][0] >0.:
                    color1 = self.prob2color(self.color2prob(img[i][j], self.source, True), self.target, True)
                    color2 = self.prob2color(self.color2prob(img[i][j], self.source, False), self.target, False)
                    #img[i][j] += mask[i][j]*np.array([color1['red'], color1['green'], color1['blue']]-img[i][j]) 
                    img[i][j] += (mask[i][j]*np.array([color1['red'], color1['green'], color1['blue']]-img[i][j]) 
                            + (1 - mask[i][j])*np.array([color2['red'], color2['green'], color2['blue']]-img[i][j])) 
                else:
                    color = self.prob2color(self.color2prob(img[i][j], self.source, False), self.target, False)
                    img[i][j] = np.array([color['red'], color['green'], color['blue']])
        return img


    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        mask = label.copy()

        img = img.astype(float)
        mask = mask.astype(float)
        if len(mask.shape) == 2:
            mask = mask[:,:,None]
            mask = np.repeat(mask,3,axis=2)
        if mask.max() > 1:
            mask /= 255
        mask_blur = self.GaussianBlur(mask, (5,5))
        img = self.GenerateImage(img, mask_blur)
        return {'image':img.astype('uint8'),
                'label': label}

if __name__ == "__main__":
    with open('./color.json') as f:
        color = json.load(f)
    with open('./non-building-color.json') as f:
        nonbuilding_color = json.load(f)
    #with open('../dataset/spacenet/building_all_cities.json') as f:
    #    data = json.load(f)
    #color_dist = color['Vegas']
    img_root = '/data/spacenet/'
    save_dir = '/data/Fake-Vegas/'
    data = os.listdir('/data/spacenet/')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Repaint = BuildingRepaint(color, nonbuilding_color)

    for name in tqdm(data):
        if 'RGB' in name:
            img = np.array(cv2.imread(img_root + name))
            gt = np.array(cv2.imread(img_root + name[:-8] + '_GT.tif'))
            sample = {'image':img[:,:,::-1], 'label':gt}
            ori = sample['image']
            sample = Repaint(sample)
            cv2.imshow("Image", sample['image'][:,:,::-1]) 
            cv2.imshow("Mask", sample['label']) 
            cv2.imshow("Original", ori[:,:,::-1]) 
            cv2.waitKey (0)
            cv2.destroyAllWindows()
            #cv2.imwrite(save_dir + name + '_RGB.tif', sample['image'].astype('uint8')[:,:,::-1])
            #cv2.imwrite(save_dir + name + '_GT.tif', sample['label'].astype('uint8')[:,:,::-1])

