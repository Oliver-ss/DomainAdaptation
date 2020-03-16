import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter

class ColorDistort(object):
    """Color distort a RGB image
    Args:
        if_pair (bool): whether to generate a copy of the original image and distort it
    """
    def __init__(self, if_pair=False):
        self.if_pair = if_pair

    def ConvertFromInts(self, image):
        return np.array(image).astype(np.float32)

    def RGB2HSV(self, image):
        image = image.astype("uint8")
        return cv2.cvtColor(image[:,:,::-1], cv2.COLOR_BGR2HSV)

    def HSV2RGB(self, image):
        image = image.astype("uint8")
        img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return img[:,:,::-1]

    def RandomSaturation(self, image, lower=0.5, upper=1.5):
        assert upper >= lower #contrast upper must be >= lower.
        assert lower >= 0 #contrast lower must be non-negative.
        image[:, :, 1] *= random.uniform(lower, upper)
        return image

    def RandomHue(self, image, delta):
        assert delta >= 0. and delta <= 180. #opencv returns H=H/2 to fit 0-255
        image[:, :, 0] += random.uniform(-delta, delta)
        image[:, :, 0][image[:, :, 0] > 180.0] -= 180.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 180.0
        return image

    def RandomContrast(self, image, lower=0.5, upper=1.5):
        assert upper >= lower #contrast upper must be >= lower.
        assert lower >= 0 #contrast lower must be non-negative.
        alpha = random.uniform(lower, upper)
        image *= alpha
        image[image > 255.0] = 255.0
        return image

    def RandomBrightness(self, image, delta=32):
        assert delta >= 0. and delta <= 255.
        delta = random.uniform(-delta, delta)
        image += delta
        image[image > 255.0] = 255.
        image[image < 0.] = 0.
        return image

    def __call__(self, sample, scale=0.5):
        image = sample['image'].copy()
        brightness = 0.2 * scale * 255.
        contrast = 0.6 * scale
        saturation = 0.8 * scale
        hue = 0.2 * scale * 180
        image = self.ConvertFromInts(image)
        image = self.RandomBrightness(image, brightness)
        image = self.RandomContrast(image, lower=1-contrast, upper=1+contrast)
        image = self.RGB2HSV(image)
        image = self.ConvertFromInts(image)
        image = self.RandomHue(image, hue)
        image = self.RandomSaturation(image, saturation)
        image = self.HSV2RGB(image)
        if self.if_pair:
            return {'image': sample['image'], 'image_pair': image, 'label': sample['label'], 'top_left': (0, 0)}
        else:
            return {'image': image, 'label': sample['label']}

class CropAndResize(object):
    """Crop a patch from the original image and resize to be the same size as the original one
    Args:
        size(tuple): the size of patch
        if_pair (bool): whether to generate a copy of the original image and distort it
    """
    def __init__(self, size=(300, 300), if_pair=False):
        self.size = size
        self.if_pair = if_pair

    def __call__(self, sample):
        image = np.array(sample['image'].copy())
        h, w = image.shape[:2]
        left_top = (random.randint(0, h-self.size[0]), random.randint(0, w-self.size[1])) #(h, w)
        left_top = (random.randint(0, h-self.size[0]), random.randint(0, w-self.size[1]))
        image_crop = image[left_top[0]:left_top[0]+self.size[0], left_top[1]:left_top[1]+self.size[1]]
        image = cv2.resize(image_crop, (w, h))
        if self.if_pair:
            return {'image': sample['image'], 'image_pair': image, 'label': sample['label'], 'top_left': left_top}
        else:
            return {'image': image, 'label': sample['label']}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), if_pair=False):
        self.mean = mean
        self.std = std
        self.if_pair = if_pair

    def __call__(self, sample):
        #print(sample)
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        mask /= 255
        mask.astype(int)
        img -= self.mean
        img /= self.std
        sample_new = {'image': img, 'label':mask}
        if self.if_pair:
            img_pair = sample['image_pair']
            img_pair = np.array(img_pair).astype(np.float32)
            img_pair /= 255
            img_pair -= self.mean
            img_pair /= self.std
            sample_new['image_pair'] = img_pair
            sample_new['top_left'] = sample['top_left']
        return sample_new


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, if_pair=False):
        self.if_pair = if_pair

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        sample_new = {'image': img, 'label':mask}
        if self.if_pair:
            img_pair = sample['image_pair']
            img_pair = np.array(img_pair).astype(np.float32).transpose((2, 0, 1))
            img_pair = torch.from_numpy(img_pair).float()
            sample_new['image_pair'] = img_pair
            sample_new['top_left'] = sample['top_left']
        return sample_new


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.8), int(self.base_size * 1.2))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}
