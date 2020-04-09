'''*************************************************************************
	> File Name: test.py
	> Author: yuansong
	> Mail: yuansongwx@outlook.com
	> Created Time: Tue 11 Feb 2020 06:07:08 PM EST
 ************************************************************************'''
from PIL import Image
from custom_transforms import PhotometricDistort, ConvertFromInts
from torchvision import transforms
import os
import cv2
import numpy as np

def transform_sample(sample):
    composed_transforms = transforms.Compose([
        ConvertFromInts(),
        PhotometricDistort(),
        ])
    return composed_transforms(sample)

names = os.listdir('../../../Sample')
for name in names:
    image = Image.open(os.path.join('../../../Sample', name)).convert('RGB')
    sample = {'image':image, 'label':None}
    sample = transform_sample(sample)
    #sample = ConvertFromInts(sample)
    #sample = PhotometricDistort(sample)
    img = sample['image'] / 255
    np.clip(img, 0., 1.)
    print(img)
    cv2.imshow("Image", img)
    cv2.waitKey (0)

