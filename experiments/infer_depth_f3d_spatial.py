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
from coco.architectures.depth import DepthPredictionScaffolder, ResidualDepth
from coco.losses import berhu_spatial
from coco.transformations import zoom_rotate, random_rgb, random_crop, normalize_images, downsample, clip, noise, exp, flip_x

global mean
mean = np.load("/data/food3d/f3d-train.npy")

def process_train(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    images, labels = flip_x(images, labels)
    images, labels = exp(images, labels)
    images, labels = zoom_rotate(images, labels)
    images, labels = random_rgb(images, labels)
    images, labels = clip(images, labels, ic=(0. ,255.))
    
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    
    images, labels = random_crop(images, labels, size)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels

def process_train_2(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    images, labels = flip_x(images, labels)
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    
    images, labels = random_crop(images, labels, size)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels


def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]
    global mean

    size = (228, 304)
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    images, labels = random_crop(images, labels, size, deterministic=True)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels


def main():
    train_db = "/data/food3d/f3d-train.hdf5"
    val_db = "/data/food3d/f3d-val.hdf5"

    batch_size = 16

    train_reader = HDF5DatabaseReader(label_key="depths")
    train_reader.setup_read(train_db)

    val_reader = HDF5DatabaseReader(label_key="depths")
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    scaffolder = DepthPredictionScaffolder(ResidualDepth, 
                                           train_processor, 
                                           val_reader=val_processor,
                                           loss=berhu_spatial)
    
    scaffolder.compile()
    scaffolder.load("/data/data/resunet.npz")
    lr_schedule = {
        1:  0.001,
        20: 0.0001
    }
    
    scaffolder.fit(40, job_name="f3d_depth_spatial_limited", snapshot="/data/data/resunet_f3d_spatial_limited.npz", momentum=0.95, lr_schedule=lr_schedule)
    scaffolder.save("/data/data/resunet_f3d_spatial_limited.npz")
    
    

if __name__ == "__main__":
    main()
