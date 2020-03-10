import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Remap(object):
    def __init__(self, table, channels):
        self.table = table
        self.channels = channels

    def __call__(self, sample):
        if random.random() < 0.5:
            img, mask = np.array(sample['image']), np.array(sample['label'].copy())
            if 'H' in self.channels: # RGB TO HSV
                img = img[:,:,::-1]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            im = np.zeros_like(img)
            for i, color in enumerate(self.channels):
                im[:,:,i] = (np.reshape(self.table[color][img[:,:,i].ravel()], img.shape[:2]))
            im = im.astype(np.uint8)
            if 'H' in self.channels: # HSV TO RGB
                im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
                im = im[:,:,::-1]
            im = Image.fromarray(im)
            return {'image':im, 'label':sample['label']}
        else:
            return sample

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        im = image.copy()
        im, _, _ = self.rand_brightness(im)
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, _, _ = distort(im)
        im, _, _ = self.rand_light_noise(im)
        return {'image': im, 'label':mask}

class ConvertFromInts(object):
    def __call__(self, sample):
        return {'image':np.array(sample['image']).astype(np.float32),
                'label':np.array(sample['label']).astype(np.float32)}

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
        return sample_new


class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img_pair = img.copy()
        img_pair = img_pair.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'image_pair': img_pair,
                'label': mask,}

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

class GaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        img_pair = sample['image_pair']
        mask = sample['label']
        img_pair = img_pair.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': img,
                'image_pair': img_pair,
                'label': mask}

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

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

class BuildingRepaint(object):
    def __init__(self, dist):
        red = np.cumsum(list(dist['red'].values()))
        green = np.cumsum(list(dist['green'].values()))
        blue = np.cumsum(list(dist['blue'].values()))
        self.dist = {'red':red, 'green':green, 'blue':blue}

    def find_prob(self, rnd, prob):
        for i, p in enumerate(prob):
            if rnd < p:
                ans = i
                break
        return i

    def choose_color(self):
        r_ = random.random()
        #g_ = random.random()
        #b_ = random.random()
        #print(r_,g_,b_)
        r = self.find_prob(r_, self.dist['red'])
        g = self.find_prob(r_, self.dist['green'])
        b = self.find_prob(r_, self.dist['blue'])
        return (r, g, b)

    def __call__(self, sample):
        if random.random() < 0.5:
            img = sample['image']
            label = sample['label']
            mask = label.copy()
            color = self.choose_color()
            #print(color)
            img = img.astype(float)
            mask = mask.astype(float)
            if len(mask.shape) == 2:
                mask = mask[:,:,None]
                mask = np.repeat(mask,3,axis=2)
            if mask.max() > 1:
                mask /= 255
            full = np.ones_like(mask).astype(float)
            im = mask.copy()
            for i, c in enumerate(color):
                im[:,:,i] *= c
            #im = im * 0.5 + (img*mask)*0.5
            im_rev = img * (full-mask)
            im += im_rev
            return {'image':im,
                    'label': label}
        else:
            return sample
