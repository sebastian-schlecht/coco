import numpy as np
import logging
from scipy.ndimage.interpolation import zoom, rotate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def exp(image, level):
    """
    Change the exposure in an RGB image
    :param img: np.array
    :param level: Number
    :return: Image

    """
    if image.dtype != np.uint8:
        logger.warn("Datatype of input image is not uint8. Are you sure that you're calling exp() on the correct array?")
    image = image.copy()

    def truncate(v):
        return 0 if v < 0 else 255 if v > 255 else v

    factor = (259. * (level + 255.)) / (255. * (259. - level))
    for x in np.nditer(image, op_flags=['readwrite']):
        x[...] = truncate(factor * (x - 128) + 128)
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


def rot_zoom_crop(image, a, f, order=0, prefilter=False):
    """
    Rotate an image according to a given angle and zoom according to a a given zoom factor
    Expectes input in axis ordering [c, h, w]
    :param image:
    :param a:
    :param f:
    :param order:
    :param prefilter:
    :return:
    """
    if len(image.shape) == 3:
        ii_r = rotate(image.transpose((1, 2, 0)), a, order=order, prefilter=prefilter)
    else:
        ii_r = rotate(image, a, order=order, prefilter=prefilter)

    h = int(image.shape[1] / f)
    w = int(image.shape[2] / f)
    s_fh = float(image.shape[1]) / float(h)
    s_fw = float(image.shape[2]) / float(w)

    cy = np.random.randint(0, image.shape[1] - h + 1)
    cx = np.random.randint(0, image.shape[2] - w + 1)

    ii_c = ii_r[cy:cy + h, cx:cx + w, :]
    if len(image.shape) == 3:
        ii_s = ii_c.transpose((2, 0, 1))
        z = (1, s_fh, s_fw)
    else:
        ii_s = ii_c
        z = (s_fh, s_fw)

    ii_s = zoom(ii_s, z, order=order, prefilter=prefilter)

    return ii_s

def mult_rgb(image, f=(1,1,1)):
    """
    Multiply color channels with a random number
    Operates in place without a copy
    """
    image[0,:,:] *= f[0]
    image[1,:,:] *= f[1]
    image[2,:,:] *= f[2]
    return image


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
