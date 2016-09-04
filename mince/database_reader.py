import h5py
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class DatabaseReader(object):
    def __init__(self, image_key="images", label_key="labels"):
        self.db = None
        self.can_read = False
        self.randomize_access = False

        self.image_key = image_key
        self.label_key = label_key

    def setup_read(self, db, randomize_access=False):
        """
        Setup the builder for db read access
        :param db:
        :param randomize_access:
        :return:
        """
        self.db = db
        self.randomize_access = randomize_access
        self.can_read = True

    def iterate(self, batch_size=8, func=None):
        """
        Single thread iteration through the data
        :param batch_size:
        :return:
        """
        raise NotImplementedError()

    def next_batch(self, batch_size=8):
        """
        Pull the next block of data out of the database
        :param batch_size:
        :return:
        """
        raise NotImplementedError()

    def db_exists(self, db):
        """
        Check whether the db at a certain path is already existing
        :param db:
        :return:
        """
        raise NotImplementedError()

    def num_samples(self):
        """
        Should return the total number of samples in the database. Usually the size of the first axis of the whole
        data block
        :return:
        """
        raise NotImplementedError()


class HDF5DatabaseReader(DatabaseReader):
    """
    Class to read images from a folder and feed a HDF5 database
    """

    def __init__(self):
        super(HDF5DatabaseReader, self).__init__()
        self.row_idx = 0
        self.db = None
        self.f = None

    def db_exists(self, db):
        # HDF5 is file based
        return os.path.isfile(db)

    def num_samples(self):
        if self.f is None:
            raise AssertionError("Please call setup_read first.")

        if self.image_key not in self.f:
            raise AssertionError("Key %s not found in database. Check your image key" % self.image_key)

        if self.label_key not inf self.f
            raise AssertionError("Key %s not found in database. Check your label key" % self.label_key)

        if self.f[self.label_key].shape[0] != self.f[self.image_key].shape[0]:
            raise AssertionError("The number of elements in the images blob does not match the number of elements in the labels blob.")

        return self.f[self.image_key].shape[0]

    def setup_read(self, db, randomize_access=False):
        super(HDF5DatabaseReader, self).setup_read(db)
        self.row_idx = 0
        self.f = h5py.File(self.db)

        self.randomize_access = randomize_access
        if self.randomize_access:
            self.permutation = np.random.permutation(self.num_samples())

    def next_batch(self, batch_size=8):
        if not self.db:
            raise AssertionError("Database not set. Please call setup_read() before calling next_batch().")

        assert self.f[self.label_key].shape[0] == self.f[self.image_key].shape[0]

        if self.row_idx + batch_size > self.f[self.label_key].shape[0]:
            self.row_idx = 0

        start_idx = self.row_idx
        self.row_idx += batch_size

        if self.randomize_access:
            perm = np.sort(self.permutation[start_idx:start_idx + batch_size]).tolist()
            excerpt = self.f[self.image_key][perm], self.f[self.label_key][perm]
        else:
            excerpt = self.f[self.image_key][start_idx:start_idx + batch_size], self.f[self.label_key][
                                                                          start_idx:start_idx + batch_size]

        return excerpt

    def iterate(self, batch_size=8, func=None):
        """
        Iterate single-threadedly
        :param batch_size: The batchsize to use
        :return:
        """
        if self.f is None:
            raise AssertionError("Please call setup_read first.")

        assert self.f[self.label_key].shape[0] == self.f[self.image_key].shape[0]

        n = self.f[self.image_key].shape[0]

        if n < batch_size:
            raise UserWarning("Batchisze %i is higher than total number of samples %i" % (batch_size, n))

        for start_idx in range(0, n - batch_size + 1, batch_size):
            data,label = self.f[self.image_key][start_idx:start_idx + batch_size], self.f[self.label_key][start_idx:start_idx + batch_size]
            if func is not None:
                data, label = func(data,label)
            yield data, label
