from __future__ import division
import cv2
import math
import torch
import random
import numbers
import numpy as np
import collections


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }


def imshow(inps, title=None):

    for idx, (inp, name) in enumerate(zip(inps, title)):
        inps[idx] = torch.clamp(inp, min=0, max=1)
        inps[idx] = inps[idx] .numpy().transpose((1, 2, 0))[:, :, ::-1]
    img = np.concatenate(inps, axis=1)
    cv2.imshow('1', img)
    cv2.waitKey(1)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
           transforms.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
         ])
    """
    __slots__ = ['transforms']

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    __slots__ = []

    def method(self, pic):
        if _is_numpy_image(pic):
            if len(pic.shape) == 2:
                pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor) or img.max() > 1:
                return img.float().div(255)
            else:
                return img
        elif _is_tensor_image(pic):
            return pic

        else:
            try:
                return self.method(np.array(pic))
            except Exception:
                raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [self.method(i) for i in pic]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    __slots__ = ['mean', 'std']

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def method(self, tensor):
        if _is_tensor_image(tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor
        elif _is_numpy_image(tensor):
            return (tensor.astype(np.float32) - 255.0 * np.array(self.mean))/np.array(self.std)
        else:
            raise RuntimeError('Undefined type')

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return [self.method(i) for i in tensor]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``
    """
    __slots__ = ['size', 'interpolation']

    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        if not (isinstance(self.size, int) or (isinstance(self.size, collections.Iterable) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

        if isinstance(self.size, int):
            h, w, c = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[self.interpolation])
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[self.interpolation])
        else:
            oh, ow = self.size
            return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[self.interpolation])

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be scaled.

        Returns:
            np.ndarray: Rescaled image.
        """
        return [self.method(i) for i in img]

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class AdjustBrightness(object):

    __slots__ = ['brightness_factor']

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.float32) * self.brightness_factor
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustContrast(object):
    __slots__ = ['contrast_factor']

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))
        im = img.astype(np.float32)
        mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
        im = (1-self.contrast_factor)*mean + self.contrast_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustSaturation(object):
    __slots__ = ['saturation_factor']

    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        im = img.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        im = (1-self.saturation_factor) * degenerate + self.saturation_factor * im
        im = im.clip(min=0, max=255)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class AdjustHue(object):
    __slots__ = ['hue_factor']

    def __init__(self, hue_factor):
        self.hue_factor = hue_factor

    def method(self, img):
        if not(-0.5 <= self.hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(self.hue_factor))

        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        im = img.astype(np.uint8)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 0] += np.uint8(self.hue_factor * 255)

        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return im.astype(img.dtype)

    def __call__(self, img):
        return [self.method(i) for i in img]


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    __slots__ = ['brightness', 'contrast', 'saturation', 'hue']

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(AdjustBrightness(brightness_factor))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(AdjustContrast(contrast_factor))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(AdjustSaturation(saturation_factor))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(AdjustHue(hue_factor))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomAffine(object):
    __slots__ = ['degrees', 'translate', 'scales', 'shears',
                 'resample', 'fillcolor', 'angle', 'translations', 'scale', 'shear']

    def __init__(self, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1),
                 shear=10, resample='BILINEAR', fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scales = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shears = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shears = shear
        else:
            self.shears = shear

        self.resample = resample
        self.fillcolor = fillcolor

    def get_params(self, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        self.angle = random.uniform(self.degrees[0], self.degrees[1])
        if self.translate is not None:
            max_dx = self.translate[0] * img_size[1]
            max_dy = self.translate[1] * img_size[0]
            self.translations = (np.round(random.uniform(-max_dx, max_dx)),
                                 np.round(random.uniform(-max_dy, max_dy)))
        else:
            self.translations = (0, 0)

        if self.scales is not None:
            self.scale = random.uniform(self.scales[0], self.scales[1])
        else:
            self.scale = 1.0

        if self.shears is not None:
            self.shear = random.uniform(self.shears[0], self.shears[1])
        else:
            self.shear = 0.0

    def method(self, img):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        assert isinstance(self.translate, (tuple, list)) and len(self.translate) == 2, \
            "Argument translate should be a list or tuple of length 2"

        assert self.scale > 0.0, "Argument scale should be positive"

        rows, cols, _ = img.shape
        center = (cols * 0.5, rows * 0.5)
        angle = math.radians(self.angle)
        shear = math.radians(self.shear)
        M00 = math.cos(angle) * self.scale
        M01 = -math.sin(angle + shear) * self.scale
        M10 = math.sin(angle) * self.scale
        M11 = math.cos(angle + shear) * self.scale
        M02 = center[0] - center[0] * M00 - center[1] * M01 + self.translate[0]
        M12 = center[1] - center[0] * M10 - center[1] * M11 + self.translate[1]
        affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12]], dtype=np.float32)
        dst_img = cv2.warpAffine(img, affine_matrix, (cols, rows), flags=INTER_MODE[self.resample],
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=self.fillcolor)
        return dst_img

    def __call__(self, img):
        self.get_params(img[0].shape)
        return [self.method(i) for i in img]


class RandomPerspective(object):
    __slots__ = ['fov_range', 'anglex_ranges', 'angley_ranges', 'anglez_ranges', 'shear_ranges',
                 'translate', 'scale_ranges', 'resample', 'fillcolor', 'fov',
                 'anglex', 'angley', 'anglez', 'shear', 'translations', 'scale']

    def __init__(self, fov=10, anglex=10, angley=10, anglez=10, shear=10,
                 translate=(0.06, 0.06), scale=(1.05, 1.05), resample='BILINEAR', fillcolor=(0, 0, 0)):

        assert all([isinstance(anglex, (tuple, list)) or anglex >= 0,
                    isinstance(angley, (tuple, list)) or angley >= 0,
                    isinstance(anglez, (tuple, list)) or anglez >= 0,
                    isinstance(shear, (tuple, list)) or shear >= 0]), \
            'All angles must be positive or tuple or list'
        assert 80 >= fov >= 0, 'fov should be in (0, 80)'
        self.fov_range = fov

        self.anglex_ranges = (-anglex, anglex) if isinstance(anglex, numbers.Number) else anglex
        self.angley_ranges = (-angley, angley) if isinstance(angley, numbers.Number) else angley
        self.anglez_ranges = (-anglez, anglez) if isinstance(anglez, numbers.Number) else anglez
        self.shear_ranges = (-shear, shear) if isinstance(shear, numbers.Number) else shear

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "translate should be a list or tuple and it must be of length 2."
        assert all([0.0 <= i <= 1.0 for i in translate]), "translation values should be between 0 and 1"
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            assert all([s > 0 for s in scale]), "scale values should be positive"
        self.scale_ranges = scale

        self.resample = resample
        self.fillcolor = fillcolor

    def get_params(self, img_size):
        """Get parameters for perspective transformation

        Returns:
            sequence: params to be passed to the perspective transformation
        """
        self.fov = 55 + random.uniform(-self.fov_range, self.fov_range)
        self.anglex = random.uniform(self.anglex_ranges[0], self.anglex_ranges[1])
        self.angley = random.uniform(self.angley_ranges[0], self.angley_ranges[1])
        self.anglez = random.uniform(self.anglez_ranges[0], self.anglez_ranges[1])
        self.shear = random.uniform(self.shear_ranges[0], self.shear_ranges[1])

        max_dx = self.translate[0] * img_size[1]
        max_dy = self.translate[1] * img_size[0]
        self.translations = (np.round(random.uniform(-max_dx, max_dx)),
                             np.round(random.uniform(-max_dy, max_dy)))

        self.scale = (random.uniform(1 / self.scale_ranges[0], self.scale_ranges[0]),
                      random.uniform(1 / self.scale_ranges[1], self.scale_ranges[1]))

    def method(self, img):
        imgtype = img.dtype
        h, w, _ = img.shape
        centery = h * 0.5
        centerx = w * 0.5

        alpha = math.radians(self.shear)
        beta = math.radians(self.anglez)

        lambda1 = self.scale[0]
        lambda2 = self.scale[1]

        tx = self.translate[0]
        ty = self.translate[1]

        sina = math.sin(alpha)
        cosa = math.cos(alpha)
        sinb = math.sin(beta)
        cosb = math.cos(beta)

        M00 = cosb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) - sinb * (lambda2 - lambda1) * sina * cosa
        M01 = - sinb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + cosb * (lambda2 - lambda1) * sina * cosa

        M10 = sinb * (lambda1 * cosa ** 2 + lambda2 * sina ** 2) + cosb * (lambda2 - lambda1) * sina * cosa
        M11 = + cosb * (lambda1 * sina ** 2 + lambda2 * cosa ** 2) + sinb * (lambda2 - lambda1) * sina * cosa
        M02 = centerx - M00 * centerx - M01 * centery + tx
        M12 = centery - M10 * centerx - M11 * centery + ty
        affine_matrix = np.array([[M00, M01, M02], [M10, M11, M12], [0, 0, 1]], dtype=np.float32)
        # -------------------------------------------------------------------------------
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(math.radians(self.fov / 2))

        radx = math.radians(self.anglex)
        rady = math.radians(self.angley)

        sinx = math.sin(radx)
        cosx = math.cos(radx)
        siny = math.sin(rady)
        cosy = math.cos(rady)

        r = np.array([[cosy, 0, -siny, 0],
                      [-siny * sinx, cosx, -sinx * cosy, 0],
                      [cosx * siny, sinx, cosx * cosy, 0],
                      [0, 0, 0, 1]])

        pcenter = np.array([centerx, centery, 0, 0], np.float32)

        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter

        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        list_dst = [dst1, dst2, dst3, dst4]

        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)

        dst = np.zeros((4, 2), np.float32)

        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

        perspective_matrix = cv2.getPerspectiveTransform(org, dst)
        total_matrix = perspective_matrix @ affine_matrix

        result_img = cv2.warpPerspective(img, total_matrix, (w, h), flags=INTER_MODE[self.resample],
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=self.fillcolor)
        return result_img.astype(imgtype)

    def __call__(self, img):
        img_shape = img[0].shape
        self.get_params(img_shape)
        return [self.method(i) for i in img]


class GaussianNoise(object):
    __slots__ = ['means', 'stds', 'gauss']

    def __init__(self, mean=0, std=0.2):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        self.means = mean
        self.stds = std

    def get_params(self, img_shape):
        mean = random.uniform(-self.means, self.means)
        std = random.uniform(0, self.stds)
        self.gauss = np.random.normal(mean, std, img_shape).astype(np.float32)

    def method(self, img):
        imgtype = img.dtype
        noisy = np.clip((1 + self.gauss) * img.astype(np.float32), 0, 255)
        return noisy.astype(imgtype)

    def __call__(self, img):
        img_shape = img[0].shape
        self.get_params(img_shape)

        return [self.method(i) for i in img]

    def __repr__(self):
        return self.__class__.__name__


class SPNoise(object):
    """Applying salt and pepper noise on the given CV Image randomly with a given probability."""
    __slots__ = ['prob', 'rnd']
    def __init__(self, prob=0.001):
        assert isinstance(prob, numbers.Number) and prob >= 0, 'p should be a positive value'
        self.prob = prob

    def method(self, img):

        imgtype = img.dtype
        noisy = img.copy()
        noisy[self.rnd < self.prob / 2] = 0.0
        noisy[self.rnd > 1 - self.prob / 2] = 255.0
        return noisy.astype(imgtype)

    def __call__(self, img):

        img_shape = img[0].shape
        self.rnd = np.random.rand(img_shape[0], img_shape[1])

        return [self.method(i) for i in img]

    def __repr__(self):
        return self.__class__.__name__


class RandomChoice(object):
    """Apply single transformation randomly picked from a list
    """
    __slots__ = ['transforms']

    def __init__(self, transforms):
        super(RandomChoice, self).__init__()
        self.transforms = transforms

    def __call__(self, img):
        return random.choice(self.transforms)(img)


class RandomOrderAndApply(object):
    __slots__ = ['transforms', 'p']
    def __init__(self, transforms, p=0.5):
        super(RandomOrderAndApply, self).__init__()
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        # random order the transforms list
        random.shuffle(self.transforms)
        # random apply transforms
        for t in self.transforms:
            if self.p < random.random():
                pass
            else:
                img = t(img)
        return img


class Pad(object):
    __slots__ = ['fill', 'padding_mode']

    def __init__(self, fill=0, padding_mode='constant'):

        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def method(self, img, padding, fill=(0, 0, 0), padding_mode='constant'):
        if not _is_numpy_image(img):
            raise TypeError('img should be CV Image. Got {}'.format(type(img)))

        if not isinstance(padding, (numbers.Number, tuple)):
            raise TypeError('Got inappropriate padding arg')
        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError('Got inappropriate fill arg')
        if not isinstance(padding_mode, str):
            raise TypeError('Got inappropriate padding_mode arg')

        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
            'Padding mode should be either constant, edge, reflect or symmetric'

        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, collections.Sequence) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, collections.Sequence) and len(padding) == 4:
            pad_left, pad_top, pad_right, pad_bottom = padding

        if isinstance(fill, numbers.Number):
            fill = fill,
        if padding_mode == 'constant':
            assert (len(fill) == 3 and len(img.shape) == 3) or (len(fill) == 1 and len(img.shape) == 2), \
                'channel of image is {} but length of fill is {}'.format(img.shape[-1], len(fill))

        img = cv2.copyMakeBorder(src=img, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                 borderType=PAD_MOD[padding_mode], value=fill)
        return img

    def __call__(self, img, padding):

        return [self.method(i, padding) for i in img]




method = [
    # Resize((200, 300)),
    # AdjustBrightness(1.9),
    # AdjustContrast(3),
    # AdjustSaturation(0.9),
    # AdjustHue(0.05),
    # ColorJitter(0.2, 0.2, 0.2, 0.1),
    # RandomAffine(),
    # RandomPerspective(),
    RandomOrderAndApply([ColorJitter(0.2, 0.2, 0.2, 0.05),
                         RandomPerspective(),
                         RandomChoice((SPNoise(), GaussianNoise()))
                         ]),
    # GaussianNoise(),
    # SPNoise(),
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ]

if __name__ == '__main__':
    for i in range(10000):
        image_path = 'E:\\personal\\draw_bbox\\dataloader\\cat.jpg'

        cvimage = cv2.imread(image_path, cv2.IMREAD_COLOR)
        imglist = [cvimage, cvimage]
        imglist = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imglist]

        methods = Compose(method)
        imglist = methods(imglist)
        sub = imglist[0] - imglist[1]
        cv2.namedWindow('1', 0)
        imshow(imglist + [sub], ('1', '2', '3'))