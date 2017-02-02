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
from coco.job import Job
from coco.architectures.depth import DepthPredictionScaffolder, ResidualDepth
from coco.transformations import zoom_rotate, random_rgb, random_crop, normalize_images, downsample, clip, noise, exp, flip_x

global mean
mean = np.load("/data/data/nyu_v2.npy")

def process_train(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)
    
    global mean
    images, labels = flip_x(images, labels)
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
    train_db = "/data/data/nyu_v2.hdf5"
    val_db = "/data/data/test_v2.hdf5"

    batch_size = 16

    train_reader = HDF5DatabaseReader(label_key="depths")
    train_reader.setup_read(train_db)

    val_reader = HDF5DatabaseReader(label_key="depths")
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    scaffolder = DepthPredictionScaffolder(ResidualDepth, train_processor, val_reader=val_processor, k=0.25)
    
    scaffolder.compile()
    Job.set_job_dir("/data/coco-jobs-relocated")
    scaffolder.fit(90, job_name="nyu_depth_qarter_thesis", snapshot="/data/data/resunet_quarter_thesis.npz", momentum=0.95,)
    scaffolder.save("/data/data/resunet_quarter_thesis.npz")
    
    

if __name__ == "__main__":
    main()
