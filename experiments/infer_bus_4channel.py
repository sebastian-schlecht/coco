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

additive_noise = np.random.normal(0, 0.001, size=(1,1,228, 304)).astype(np.float32)

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
    
    # Add the noise channel
    images = np.concatenate([images, additive_noise.repeat(images.shape[0], axis=0)], axis=1)
    
    
    return images, labels


def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]
    global mean

    size = (228, 304)
    images, _ = normalize_images(images, labels, mean, std=71.571201304890508)
    images, _ = random_crop(images, None, size, deterministic=True)
    
    # Add the noise channel
    images = np.concatenate([images, additive_noise.repeat(images.shape[0], axis=0)], axis=1)

    return images, labels


def main():
    train_db = "/data/food3d/f3d-labels-f3-train.hdf5"
    val_db = "/data/food3d/f3d-labels-f3-val.hdf5"

    batch_size = 32

    train_reader = HDF5DatabaseReader(label_key="bus")
    train_reader.setup_read(train_db, randomize_access=True)

    val_reader = HDF5DatabaseReader(label_key="bus")
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    
    scaffolder = BURegressionScaffolder(BURegressor, 
                                           train_reader=train_processor, 
                                           val_reader=val_processor,
                                           with_depth=True)
    
    lr_schedule = {
        1: 0.0001,
        20: 0.0001,
        38: 0.00001
    }
    scaffolder.load("/data/data/resnet50-food-101.npz", strict=False)
    scaffolder.compile()
    out_file = "/data/data/bu_regressor_4channel_f3.npz"
    scaffolder.fit(40, job_name="bu_regression_4channel_f3", snapshot=out_file, momentum=0.95, lr_schedule=lr_schedule)
    scaffolder.save(out_file)
    
    

if __name__ == "__main__":
    main()
