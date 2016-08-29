import h5py
import os
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ClassDatabaseBuilder(object):
    def build_read(self, db, folder, shape=None, force=False):
        """
        Parse a folder, build the db and return an iterator object
        Re-use the db if already existing
        TODO: Implement global shuffling
        """
        if not self.db_exists(db) or force:
            try:
                os.remove(db)
            except:
                pass
            self.build_db(db, folder, shape)
        self.db = db

    def iterate(batch_size=8):
        raise NotImplementedError()

    def build_db(self, db, folder, shape=None):
        if shape is None:
            raise AssertionError("Parameter shape has to be either an int or a tuple but cannot be None.")
        raise NotImplementedError()

    def db_exists(db):
        raise NotImplementedError()

    def get_db_iterator(self, db):
        raise NotImplementedError()

    def parse_folder(self, folder_name):
        """
        Parse a folder of images and read a structure of files
        """
        classes = {}
        for _dir in [o for o in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name,o))]:
            # List files
            classes[_dir] = []
            for f in [f for f in os.listdir(os.path.join(folder_name, _dir)) if os.path.isfile(os.path.join(folder_name,_dir,f))]:
                p = os.path.join(folder_name, _dir, f)
                if self.is_image(p):
                    classes[_dir].append(p)
        return classes


    def is_image(self, image_path):
        p = image_path
        return p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".JPG") or p.endswith(".png")

    def read_image(self, image_path, resize=None):
        """
        Read a single image into a numpy memory
        """
        image_obj = Image.open(image_path)
        if resize:
            image_obj = image_obj.resize(resize)

        return np.array(image_obj).transpose((2,0,1))




class HDF5ClassDatabaseBuilder(ClassDatabaseBuilder):
    def __init__(self):
        super(HDF5ClassDatabaseBuilder, self).__init__()

    def db_exists(self, db):
        return os.path.isfile(db)

    def get_db_iterator(self, db):
        f = h5py.File(db)

    def iterate(self, batch_size=8):
        """
        Generator to walk data
        """
        if self.db is None:
            raise AssertionError("Please call build_read first to assign a DB")

        f = h5py.File(self.db)
        assert f["labels"].shape[0] == f["images"].shape[0]

        n = f["images"].shape[0]

        for start_idx in range(0, n - batch_size + 1, batch_size):
            yield f["images"][start_idx:start_idx+batch_size], f["labels"][start_idx:start_idx+batch_size],


    def build_db(self, db, folder, shape):
        """
        Parse RGB images and build a db out of it
        TODO:
        Support grayscale
        """
        logger.info("Start building database")
        classes = self.parse_folder(folder)
        f = h5py.File(db)

        classes = self.parse_folder(folder)
        num_classes = len(classes)

        image_ds = f.create_dataset('images', (0,0,0,0), maxshape=(None,None, None, None ), dtype=np.uint8)
        label_ds = f.create_dataset('labels', (0,0), maxshape=(None,None ), dtype=np.uint8)

        label_idx = 0
        row_ptr = 0
        for cls in classes:
            logger.debug("Processing class %s" % cls)
            # Read images
            images = None
            labels = None
            for image_path in classes[cls]:
                image_array = np.expand_dims(self.read_image(image_path, resize=shape).copy(), 0)
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
            assert(image_shape[0] == label_shape[0])
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
