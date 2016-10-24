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
        logger.warn(
            "Datatype of input image is not uint8.")
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
    
def mult_rgb(image, f=(1, 1, 1)):
    """
    Multiply color channels with a random number
    Operates in place without a copy
    """
    image[0, :, :] *= f[0]
    image[1, :, :] *= f[1]
    image[2, :, :] *= f[2]
    return image

def crop(image, center ,shape):
    h, w = shape
    cy, cx = center
    if len(image.shape) == 3:
        return image[:, cy:cy+h, cx:cx+w]
    else:
        return image[cy:cy+h, cx:cx+w]



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



def zoom_rot(ii,dd):
    """
    Special case. Transform data and labels in conjunction
    """
    a = np.random.randint(-10,10)
    ddr = rotate(dd,a, order=0, prefilter=False)
    iir = rotate(ii.transpose((1,2,0)),a, order=2)

    f = np.random.randint(10000,15100) / 10000.

    h = int(dd.shape[0] / f)
    w = int(dd.shape[1] / f)

    s_fh = float(dd.shape[0]) / float(h)
    s_fw = float(dd.shape[1]) / float(w)

    s_f = (s_fh + s_fw) / 2.

    offset  = 0
    cy = np.random.randint(offset,dd.shape[0] - h - offset + 1)
    cx = np.random.randint(offset,dd.shape[1] - w - offset + 1)

    ddc = ddr[cy:cy+h, cx:cx+w]
    iic = iir[cy:cy+h,cx:cx+w,:]

    dd_s = zoom(ddc,(s_fh, s_fw),order=0, prefilter=False)
    dd_s /= s_f
    ii_s = iic.transpose((2,0,1))

    ii_s = zoom(ii_s,(1,s_fh,s_fw),order=2)

    return ii_s.astype(np.float32), dd_s.astype(np.float32)
