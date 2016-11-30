import numpy as np

import coco.augmentations as A


def zoom_rotate(images, labels, deterministic=False):
    """
    Zoom and rotate images and labels in conjunction
    :param images:
    :param labels:
    :param deterministic:
    :return:
    """
    if deterministic:
        return images, labels
    for i in range(images.shape[0]):
        # Zoom and rotate
        p = np.random.randint(2)
        if p > 0:
            images[i], labels[i] = A.zoom_rot(images[i], labels[i])
    return images, labels


def flip_x(images, labels, deterministic=False):
    """
    Flip images along x axis
    :param images:
    :param labels:
    :param deterministic:
    :return:
    """
    if deterministic:
        return images, labels

    for i in range(images.shape[0]):
        # Flip
        p = np.random.randint(2)
        if p > 0:
            images[i] = A.flip_x(images[i])
            if labels is not None:
                labels[i] = A.flip_x(labels[i])
    return images, labels


def exp(images, labels, deterministic=False):
    """
    Change exposure of the images
    :param images:
    :param labels:
    :param deterministic:
    :return:
    """
    if deterministic:
        return images, labels
    for i in range(images.shape[0]):
        lvl = np.random.randint(0, 10)
        images[i] = A.exp(images[i], lvl)
    return images, labels


def random_rgb(images, labels, deterministic=False):
    """
    Scale pixels with random RGB values
    :param images:
    :param labels:
    :param deterministic:
    :return:
    """
    if deterministic:
        return images, labels
    for i in range(images.shape[0]):
        # Random RGB
        r = np.random.randint(85, 116) / 100.
        g = np.random.randint(85, 116) / 100.
        b = np.random.randint(85, 116) / 100.
        images[i] = A.mult_rgb(images[i], f=(r, g, b))
    return images, labels


def noise(images, labels, deterministic=False):
    """
    Add gaussian noise to the image
    :param images:
    :param labels:
    :param deterministic:
    :return:
    """
    if deterministic:
        return images, labels
    for i in range(images.shape[0]):
        images[i] = A.add_noise(images[i])
    return images, labels


def normalize_images(images, labels, mean, std=None):
    """
    Normalize images by subtracting mean and dividing be their std
    :param images:
    :param labels:
    :param mean:
    :param std:
    :return:
    """
    for i in range(images.shape[0]):
        images[i] -= mean
        if std:
            images[i] /= std
    return images, labels


def clip(images, labels, ic=None, lc=None):
    """
    Clip images. Mostly to shift values in meaningful range
    :param images:
    :param labels:
    :param ic:
    :param lc:
    :return:
    """
    if ic:
        images = images.clip(ic[0], ic[1])
    if lc:
        labels = labels.clip(lc[0], lc[1])
    return images, labels


def downsample(images, labels, factors):
    """
    Downsample images numpy style. X and Y along the same factor
    :param images:
    :param labels:
    :param factors:
    :return:
    """
    if len(images.shape) == 4:
        i = images[:, :, ::factors[0], ::factors[0]]
    else:
        i = images[:, ::factors[0], ::factors[0]]

    if len(labels.shape) == 4:
        l = labels[:, :, ::factors[1], ::factors[1]]
    else:
        l = labels[:, ::factors[1], ::factors[1]]
    return i, l


def random_crop(images, labels, size, deterministic=False):
    """
    Crop images randomly
    :param images:
    :param labels:
    :param size:
    :param deterministic:
    :return:
    """
    h, w = size
    new_image_shape = list(images.shape)
    new_image_shape[-2] = h
    new_image_shape[-1] = w
    new_images = np.zeros(new_image_shape, dtype=np.float32)
    
    if labels is not None:
        new_label_shape = list(labels.shape)
        new_label_shape[-2] = h
        new_label_shape[-1] = w
        new_labels = np.zeros(new_label_shape, dtype=np.float32)

    
    if deterministic:
        for i in range(images.shape[0]):
            cy = (images.shape[2] - h) // 2
            cx = (images.shape[3] - w) // 2
            new_images[i] = A.crop(images[i], (cy, cx), (h, w))
            if labels is not None:
                new_labels[i] = A.crop(labels[i], (cy, cx), (h, w))
    else:
        for i in range(images.shape[0]):
            cy = np.random.randint(images.shape[2] - h)
            cx = np.random.randint(images.shape[3] - w)
            new_images[i] = A.crop(images[i], (cy, cx), (h, w))
            if labels is not None:
                new_labels[i] = A.crop(labels[i], (cy, cx), (h, w))

    if labels is not None:
        return new_images, new_labels
    else:
        return new_images, labels
