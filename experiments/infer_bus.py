import sys
import os
import inspect

import numpy as np

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from coco.database_reader import HDF5DatabaseReader
from coco.multiprocess import MultiProcessor
from coco.architectures.regression import BURegressionScaffolder, BURegressor
from coco.transformations import zoom_rotate, random_rgb, random_crop, normalize_images, downsample, clip, noise, exp, flip_x

global mean
mean = np.load("/data/food3d/f3d-train.npy")

def process_train(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    # Loop labels through but return the copy. Not efficient but we don't need to augment the training labels here
    labels_copy = labels.copy()
    images, labels = flip_x(images, labels)
    images, labels = exp(images, labels)
    images, labels = zoom_rotate(images, labels)
    images, labels = random_rgb(images, labels)
    images, labels = clip(images, labels, ic=(0. ,255.))
    
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    
    images, labels = random_crop(images, labels, size)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels_copy


def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]
    global mean
    # Same as in process_train
    labels_copy = labels.copy()

    size = (228, 304)
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    images, labels = random_crop(images, labels, size, deterministic=True)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels_copy


def main():
    train_db = "/data/food3d/f3d-labels-train.hdf5"
    val_db = "/data/food3d/f3d-labels-val.hdf5"

    batch_size = 16

    train_reader = HDF5DatabaseReader(label_key="depths")
    train_reader.setup_read(train_db)

    val_reader = HDF5DatabaseReader(label_key="bus")
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    
    scaffolder = BURegressionScaffolder(BURegressor, 
                                           train_reader=train_processor, 
                                           val_reader=val_processor)
    scaffolder.compile()
    out_file = "/data/data/bu_regressor.npz"
    scaffolder.fit(40, job_name="bu_regression", snapshot=out_file, momentum=0.9)
    scaffolder.save(out_file)
    
    

if __name__ == "__main__":
    main()
