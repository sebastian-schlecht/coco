import h5py
import os
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ClassDatabaseBuilder(object):
    def __init(self):
        self.db = None

    def build(self, db, folder, shape=None, force=False):
        """
        Build a db from a a folder
        :param db:
        :param folder:
        :param shape:
        :param force:
        :return:
        """
        if not self.db_exists(db) or force:
            try:
                os.remove(db)
            except:
                pass
            self.build_db(db, folder, shape)

    def setup_read(self, db):
        """
        Setup the builder for db read access
        :param db:
        :return:
        """
        self.db = db

    def iterate(self, batch_size=8):
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

    def build_db(self, db, folder, shape=None):
        """
        Build a database to a file called db, from a folder called folder and resize to shape
        :param db:
        :param folder:
        :param shape:
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

    @staticmethod
    def parse_folder(folder_name):
        """
        Parse a folder of images and read a structure of files
        :param folder_name:
        :return:
        """
        classes = {}
        for _dir in [o for o in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, o))]:
            # List files
            classes[_dir] = []
            for f in [f for f in os.listdir(os.path.join(folder_name, _dir)) if
                      os.path.isfile(os.path.join(folder_name, _dir, f))]:
                p = os.path.join(folder_name, _dir, f)
                if ClassDatabaseBuilder.is_image(p):
                    classes[_dir].append(p)
        return classes

    @staticmethod
    def is_image(image_path):
        """
        Check if a file is an image
        :param image_path:
        :return:
        """
        p = image_path
        return p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".JPG") or p.endswith(".png")

    @staticmethod
    def read_image(image_path, resize=None):
        """
        Read a single image into a numpy memory
        :param image_path:
        :param resize:
        :return:
        """
        image_obj = Image.open(image_path)
        if resize:
            image_obj = image_obj.resize(resize)

        return np.array(image_obj).transpose((2, 0, 1))


class HDF5ClassDatabaseBuilder(ClassDatabaseBuilder):
    """
    Class to read images from a folder and feed a HDF5 database
    """
    def __init__(self):
        super(HDF5ClassDatabaseBuilder, self).__init__()
        self.row_idx = 0
        self.db = None
        self.f = None

    def db_exists(self, db):
        # HDF5 is file based
        return os.path.isfile(db)

    def num_samples(self):
        if self.f is None:
            raise AssertionError("Please call setup_read first.")

        assert self.f["labels"].shape[0] == self.f["images"].shape[0]

        return self.f["images"].shape[0]

    def setup_read(self, db):
        super(HDF5ClassDatabaseBuilder, self).setup_read(db)
        self.row_idx = 0
        if self.db is None:
            raise AssertionError("Please call build first to assign a DB")
        self.f = h5py.File(self.db)

    def next_batch(self, batch_size=8):
        if not self.db:
            raise AssertionError("DB not set. Please call setup_read() before calling next_batch()")

        assert self.f["labels"].shape[0] == self.f["images"].shape[0]

        if self.row_idx + batch_size > self.f["labels"].shape[0]:
            self.row_idx = 0

        start_idx = self.row_idx
        self.row_idx += batch_size

        return self.f["images"][start_idx:start_idx + batch_size], self.f["labels"][start_idx:start_idx + batch_size]

    def iterate(self, batch_size=8):
        """
        Iterate single-threadedly
        :param batch_size: The batchsize to use
        :return:
        """
        if self.f is None:
            raise AssertionError("Please call setup_read first.")

        assert self.f["labels"].shape[0] == self.f["images"].shape[0]

        n = self.f["images"].shape[0]

        for start_idx in range(0, n - batch_size + 1, batch_size):
            yield self.f["images"][start_idx:start_idx + batch_size], self.f["labels"][start_idx:start_idx + batch_size]

    def build_db(self, db, folder, shape=None):
        """
        Parse a folder and build the HDF5 database
        TODO Support grayscale
        :param db: The target db file to create
        :param folder: The folder to parse
        :param shape: Resize shape
        :return:
        """
        logger.info("Start building database")
        f = h5py.File(db)

        classes = HDF5ClassDatabaseBuilder.parse_folder(folder)
        num_classes = len(classes)

        image_ds = f.create_dataset('images', (0, 0, 0, 0), maxshape=(None, None, None, None), dtype=np.uint8)
        label_ds = f.create_dataset('labels', (0, 0), maxshape=(None, None), dtype=np.uint8)

        label_idx = 0
        row_ptr = 0
        for cls in classes:
            logger.debug("Processing class %s" % cls)
            # Read images
            images = None
            labels = None
            for image_path in classes[cls]:
                image_array = np.expand_dims(HDF5ClassDatabaseBuilder.read_image(image_path, resize=shape).copy(), 0)
                if images is None:
                    images = image_array
                else:
                    images = np.concatenate([images, image_array])

                label = np.zeros(num_classes)
                label[label_idx] = 1
                label = np.expand_dims(label, 0)

                if labels is None:
                    labels = label
                else:
                    labels = np.concatenate([labels, label])

            image_shape = image_ds.shape
            label_shape = label_ds.shape
            assert (image_shape[0] == label_shape[0])
            if label_idx == 0:
                image_ds.resize(images.shape)
                image_ds[:] = images

                label_ds.resize(labels.shape)
                label_ds[:] = labels
                row_ptr += images.shape[0]

            # Append
            else:
                image_ds.resize(image_shape[0] + images.shape[0], axis=0)
                image_ds[row_ptr:] = images

                label_ds.resize(label_shape[0] + labels.shape[0], axis=0)
                label_ds[row_ptr:] = labels

            label_idx += 1

        f.close()
