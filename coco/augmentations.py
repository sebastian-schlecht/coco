import numpy as np
import logging
from scipy.ndimage.interpolation import zoom, rotate
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def exp(image, level):
    """
    Change the exposure in an RGB image
    :param img: np.array
    :param level: Number
    :return: Image
    """
    image = image.copy().astype(np.float32)
    factor = (259. * (level + 255.)) / (255. * (259. - level))
    image = factor * (image - 128) + 128
    image = image.clip(0., 255.)
    return image


def flip_x(image):
    """
    Flip image along x axis
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        return image[:, :, ::-1]
    else:
        return image[:, ::-1]


def flip_y(image):
    """
    Flip image along y axis
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        return image[:, ::-1, :]
    else:
        return image[::-1, :]


def mult_rgb(image, f=(1, 1, 1)):
    """
    Multiply color channels with a random number
    Operates in place without a copy
    """
    image[0, :, :] *= f[0]
    image[1, :, :] *= f[1]
    image[2, :, :] *= f[2]
    return image


def crop(image, center, shape):
    """
    Crop an image
    :param image:
    :param center:
    :param shape:
    :return:
    """
    h, w = shape
    cy, cx = center
    if len(image.shape) == 3:
        return image[:, cy:cy + h, cx:cx + w]
    else:
        return image[cy:cy + h, cx:cx + w]


def pad_crop(image, padsize=4):
    """
    Pad an image with 0s and do a random crop
    :param image:
    :param padsize:
    :return:
    """
    cx = np.random.randint(2 * padsize)
    cy = np.random.randint(2 * padsize)

    if len(image.shape) == 3:
        padded = np.pad(image, ((0, 0), (padsize, padsize), (padsize, padsize)))
        x = image.shape[2]
        y = image.shape[1]
        return padded[:, cy:cy + y, cx:cx + x]
    else:
        padded = np.pad(image, ((padsize, padsize), (padsize, padsize)))
        x = image.shape[1]
        y = image.shape[0]
        return padded[cy:cy + y, cx:cx + x]


def add_noise(image, strength=0.2, mu=0, sigma=50):
    """
    Add some noise to the image
    :param image:
    :param strength:
    :param mu:
    :param sigma:
    :return:
    """
    noise = np.random.normal(mu, sigma, size=image.shape)
    noisy = image + strength * noise
    return noisy


def zoom_rot(ii, dd):
    """
    Special case. Transform data and labels in conjunction
    :param ii:
    :param dd:
    :return:
    """
    a = np.random.randint(-10, 10)
    ddr = rotate(dd, a, order=0, prefilter=False, reshape=False)
    iir = rotate(ii.transpose((1, 2, 0)), a, order=2, reshape=False)

    rads = (abs(a) / 180.) * math.pi

    h = ii.shape[1]
    w = ii.shape[2]
    x = math.ceil(math.tan(rads) * (h / 2.))
    y = math.ceil(math.tan(rads) * (w / 2.))

    max_w = w - 2 * x
    max_h = h - 2 * y

    min_f = max(w / max_w, h / max_h)
    f = np.random.uniform(min_f, 1.5)

    n_h = int(h / f)
    n_w = int(w / f)

    upper = h - n_h - y
    cy = np.random.randint(y, upper)
    upper = w - n_w - x
    cx = np.random.randint(x, upper)

    ddc = ddr[cy:cy + n_h, cx:cx + n_w]
    iic = iir[cy:cy + n_h, cx:cx + n_w, :]

    s_fh = float(h) / float(n_h)
    s_fw = float(w) / float(n_w)

    dd_s = zoom(ddc, (s_fh, s_fw), order=0, prefilter=False)
    dd_s /= f

    ii_s = iic.transpose((2, 0, 1))
    ii_s = zoom(ii_s, (1, s_fh, s_fw), order=2)

    return ii_s.astype(np.float32), dd_s.astype(np.float32)
