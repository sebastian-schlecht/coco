import h5py
import os
from PIL import Image
import numpy as np
import logging
import collections
from multiprocess import Process
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ClassDatabaseBuilder(object):
    def __init__(self):
        raise Exception("This class is purely static.")

    @classmethod
    def build(cls, db, folder, shape=None, partition=(0.7, 0.3, None)):
        """
        Build a database to a file called db, from a folder called folder and resize to shape
        :param db:
        :param folder:
        :param shape:
        :param partition:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def db_exists(cls, db):
        """
        Check whether the db at a certain path is already existing
        :param db:
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def parse_folder(cls, folder_name):
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

    @classmethod
    def is_image(cls, image_path):
        """
        Check if a file is an image
        :param image_path:
        :return:
        """
        p = image_path
        return p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".JPG") or p.endswith(".png")

    @classmethod
    def read_image(cls, image_path, resize=None):
        """
        Read a single image into a numpy memory
        :param image_path:
        :param resize:
        :return:
        """
        image_obj = Image.open(image_path)
        if resize:
            image_obj = image_obj.resize(resize)
        
        image = np.array(image_obj)
        if len(image.shape) == 3:
            return image.transpose((2, 0, 1))
        else:
            return image


class HDF5ClassDatabaseBuilder(ClassDatabaseBuilder):
    """
    Class to read images from a folder and feed a HDF5 database
    """
    @classmethod
    def db_exists(cls, db):
        # HDF5 is file based
        return os.path.isfile(db)

    @classmethod
    def _build_db_for_file_list(cls, db_file, file_list, shape):

        BLOCK_SIZE = 10

        logger.info("Start building database %s" % db_file)
        f = h5py.File(db_file)
        classes = file_list
        num_classes = len(classes)

        image_ds = f.create_dataset('images', (0, 0, 0, 0), 
                                    maxshape=(None, None, None, None), 
                                    dtype=np.uint8, 
                                    chunks=(BLOCK_SIZE, 3 , shape[0], shape[1]))
        label_ds = f.create_dataset('labels', (0, 0), 
                                    maxshape=(None, None), 
                                    dtype=np.uint8,
                                    chunks=(BLOCK_SIZE, num_classes))
        label_idx = 0
        row_ptr = 0
        od = collections.OrderedDict(sorted(classes.items()))

        item_list = []
        for cls, v in od.iteritems():
            label = label_idx
            for image_path in classes[cls]:
                entry = (label, image_path)
                item_list.append(entry)
            label_idx += 1

        random.shuffle(item_list)
        BLOCK_SIZE = min(BLOCK_SIZE, len(item_list))
        for index in range(0,len(item_list) - BLOCK_SIZE + 1, BLOCK_SIZE):
            logger.debug("Processing block %i out of %i" % (index, len(item_list) - BLOCK_SIZE + 1))
            block = item_list[index:index + BLOCK_SIZE]
            # Read images
            images = None
            labels = None
            for label_idx, image_path in block:
                image_array = np.expand_dims(HDF5ClassDatabaseBuilder.read_image(image_path, resize=shape).copy(), 0)
                   
                if image_array is None or len(image_array.shape) != 4:
                    continue
                
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
            if index == 0:
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
                row_ptr += images.shape[0]
            label_idx += 1
        f.close()

    @classmethod
    def build(cls, db, folder, shape=None, partition=(0.7, 0.3, None), force=False):
        """
        Parse a folder and build the HDF5 database
        TODO Support grayscale
        :param db: The target db file to create
        :param folder: The folder to parse
        :param shape: Resize shape
        :param partition: Partition of the dataset
        :return:
        """
        classes = cls.parse_folder(folder)
        PARTS = ["train", "val", "test"]

        if len(partition) > 3:
            raise AssertionError(
                "Currently only 3 types of databases are supported: train, val & test. Thus the partition array must be max 3 items")
        sum = 0
        for p in partition:
            if p is not None:
                sum += p

        if sum != 1.: raise AssertionError("Sum of all partitions must be equal to 1")

        results = {}
        for i in range(len(partition)): results[PARTS[i]] = {}

        for key in classes:
            _list = classes[key]
            _len = len(_list)
            idx = 0
            for i in range(len(partition)):
                frac = partition[i]
                if frac is None:
                    continue
                n = int(frac * _len)
                new_list = _list[idx:idx+n]
                idx += n
                results[PARTS[i]][key] = new_list

        files = []
        processes = []
        for index in range(len(partition)):
            filename = "%s-%s.h5" % (db, PARTS[index])
            if partition[index]:
                files.append(filename)

            if cls.db_exists(filename):
                if force:
                    try:
                        os.remove(filename)
                    except:
                        pass
                else:
                    continue

            classes = results[PARTS[index]]
            if len(classes) > 0:
                p = Process(target=cls._build_db_for_file_list, args=(filename, classes, shape))
                p.daemon = True
                p.start()
                processes.append(p)
        for p in processes:
            p.join()


        return files
