import unittest
import numpy as np

from mince.augmentations import flip_x, flip_y, pad_crop


class TestAugmentations(unittest.TestCase):
    def test_flip_x(self):
        a = np.arange(100).reshape((10, 10))
        b = a[:, ::-1]

        self.assertTrue(np.array_equal(flip_x(a), b))

        a = np.arange(300).reshape((3, 10, 10))
        b = a[:, :, ::-1]

        self.assertTrue(np.array_equal(flip_x(a), b))

    def test_flip_y(self):
        a = np.arange(100).reshape((10, 10))
        b = a[::-1, :]

        self.assertTrue(np.array_equal(flip_y(a), b))

        a = np.arange(300).reshape((3, 10, 10))
        b = a[:, ::-1, :]

        self.assertTrue(np.array_equal(flip_y(a), b))
