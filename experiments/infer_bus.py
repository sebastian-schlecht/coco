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
from coco.transformations import random_rgb, random_crop, normalize_images, downsample, clip, noise, exp, flip_x

global mean
mean = np.load("/data/food3d/f3d-train.npy")

def process_train(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    
    images, _ = exp(images, labels)
    images, _ = random_rgb(images, labels)
    images, _ = clip(images, labels, ic=(0. ,255.))
  
    images, _ = normalize_images(images, labels, mean, std=71.571201304890508)
    
    images, _ = random_crop(images, None, size)
    
    return images, labels


def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]
    global mean

    size = (228, 304)
    images, _ = normalize_images(images, labels, mean, std=71.571201304890508)
    images, _ = random_crop(images, None, size, deterministic=True)

    return images, labels


def main():
    train_db = "/data/food3d/f3d-labels-train.hdf5"
    val_db = "/data/food3d/f3d-labels-val.hdf5"

    batch_size = 16

    train_reader = HDF5DatabaseReader(label_key="bus")
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
