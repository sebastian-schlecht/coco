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
from coco.job import Job

global mean
mean = np.load("/ssd/food3d/f3d-train.npy")

def process_train(images, labels):
    images = images.astype(np.float32)
    images = images[:,0:3,:,:]
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    
    images, _ = normalize_images(images, labels, mean)
    images, _ = random_crop(images, None, size)
    
    return images, labels


def process_val(images, labels):
    images = images.astype(np.float32)
    images = images[:,0:3,:,:]
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]
    
    size = (228, 304)
    
    global mean

    images, _ = normalize_images(images, labels, mean)
    images, _ = flip_x(images, None)
    images, _ = random_crop(images, None, size, deterministic=True)

    return images, labels


def main():
    train_db = "/ssd/food3d/f3d-rgbd-train.hdf5"
    val_db = "/ssd/food3d/f3d-rgbd-val.hdf5"

    batch_size = 32

    train_reader = HDF5DatabaseReader(label_key="bus", image_key="rgbd")
    train_reader.setup_read(train_db)

    val_reader = HDF5DatabaseReader(label_key="bus", image_key="rgbd")
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    Job.set_job_dir("/data/coco-jobs-relocated")
    scaffolder = BURegressionScaffolder(BURegressor, 
                                           train_reader=train_processor, 
                                           val_reader=val_processor,
                                           with_depth=False)
    
    scaffolder.load("/data/data/resnet50-food-101.npz", strict=False)
    scaffolder.compile()
    out_file = "/data/data/bu_regressor_rgb_only_f1_thesis.npz"
    scaffolder.fit(40, job_name="bu_regression_rgb_only_thesis_f1", snapshot=out_file, momentum=0.95)
    scaffolder.save(out_file)
    
    

if __name__ == "__main__":
    main()
