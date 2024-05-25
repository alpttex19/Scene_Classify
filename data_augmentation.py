import cv2
import numpy as np
import torch
from numpy import random

# ----------------------- Augmentation Functions -----------------------

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

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class Resize(object):
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, image):
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        return image


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        image = image.astype(np.float64)
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        image = image.astype(np.float64)
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image



class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        image = image.astype(np.float64)
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        image = image.astype(np.float64)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
    """
    def __call__(self, image):
        image = image.astype(np.float64)
        height, width, _ = image.shape
        # max trails (50)
        for _ in range(5):
            current_image = image

            w = random.uniform(0.3 * width, width)
            h = random.uniform(0.3 * height, height)

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            left = random.uniform(width - w)
            top = random.uniform(height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            # cut the crop from the image
            current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]

            return current_image
        return image


class Expand(object):
    def __call__(self, image):
        image = image.astype(np.float64)
        if random.randint(2):
            return image

        height, width, depth = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)

        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        return image


class RandomHorizontalFlip(object):
    def __call__(self, image):
        if random.randint(2):
            image = image[:, ::-1]

        return image
    

class RandomVerticalFlip(object):
    def __call__(self, image):
        if random.randint(2):
            image = image[::-1, :]

        return image
    
class RandomRotate(object):
    def __init__(self, angle=90):
        self.angle = random.uniform(-angle, angle)

    def __call__(self, image):
        image = image.astype(np.float64)
        angle = random.uniform(-self.angle, self.angle)
        h, w, _ = image.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

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

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        return im


# ----------------------- Main Functions -----------------------
## SSD-style Augmentation
class Augmentation(object):
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.augment = Compose([
            RandomSaturation(),                        # 随机饱和度
            RandomHue(),                               # 随机色调
            RandomContrast(),                          # 随机对比度
            RandomBrightness(),                         # 随机亮度
            RandomSampleCrop(),                        # 随机剪裁
            RandomHorizontalFlip(),                    # 随机水平翻转
            RandomRotate(angle=10),                   # 随机旋转
            Resize(self.img_size)                      # resize操作
        ])

    def __call__(self, image):
        img =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # augment
        image = self.augment(img)
        # to tensor
        img_tensor = (torch.from_numpy(image).permute(2,0,1)) / 255        

        return img_tensor
    

## SSD-style valTransform
class BaseTransform(object):
    def __call__(self, image): 
        # to tensor
        img_tensor = (torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1)) / 255 
        return img_tensor
