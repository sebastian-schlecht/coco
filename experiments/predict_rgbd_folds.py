"""
In order to compare inferred and GT depth together with RGB input for BU regression, we'd like to make the comparison as fair as possible. Thus, we compile a dataset for the inferred case where the depth is predicted by a network that has never seen that particular sample before during its training phase. We thus compile a dataset with inferred depth based on the learned folds.
"""
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
from coco.transformations import zoom_rotate, random_rgb, random_crop, normalize_images, downsample, clip, noise, exp, flip_x
from coco.job import Job

from scipy.ndimage.interpolation import zoom

global mean
mean = np.load("/ssd/food3d/f3d-train.npy")

def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)
    
    # Extract data from combined blob
    images = images[:,0:3,:,:]
    labels = labels[:,3,:,:]
    
    assert images.shape[0] == labels.shape[0]
    global mean

    size = (228, 304)
    images, labels = normalize_images(images, labels, mean, std=71.571201304890508)
    images, labels = random_crop(images, labels, size, deterministic=True)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels


def main():
    db_models = [
        ("/ssd/food3d/f3d-rgbd-val.hdf5", "/data/data/resunet_f3d_half_limited_f1_thesis.npz")
    ]
    
    
    print "Predicting depth"
    
    ds_images = []
    ds_depths = []
    ds_bus = []
    ds_labels = []
    
    for train, val, model in db_models:
        print "Predicting depth for db: %s" % train
        scaffolder = DepthPredictionScaffolder(ResidualDepth, inference=True, k=0.5)
        scaffolder.compile()
        scaffolder.load(model)
        
        f = h5py.File(val)
        ilen = f["rgbd"].shape[0]
        
        for index in range(ilen):
            image = np.array(f["rgbd"][index])[np.newaxis,:,:,:]
            # Just copy, separation is done by processing function
            label = image.copy()
            
            # Preprocess
            image, label = process_val(image, label)
            
            # Predict
            prediction = scaffolder.infer(image)[0]
            
            # Post process
            upper = np.percentile(prediction, 99)
            lower = np.percentile(prediction, 1)
            prediction = prediction.clip(lower, upper)
            
            # Up-sample
            prediction = predicition.squeeze()
            f_h = 240. / prediction.shape[0]
            f_w = 320. / prediction.shape[1]
            prediction = zoom(prediction, (f_h, f_w))
            # Add num and channel axis again
            prediction = prediction[np.newaxis, np.newaxis, :, :]
            
            # Store
            ds_images.append(image)
            ds_depths.append(prediction)
            ds_bus.append(f["bus"][index])
            ds_labels.append(f["labels"][index])              
        
        f.close()
                          
    # We want to provide similar conditions to both nets so we iterate the train dbs and find all corresponding dishes
    # The result will contain every image twice but that's ok as they would be re-iterated anyways after the next epoch
    for (train, val, model) in db_models:
        print "Building fold for db: %s" % train
        f = h5py.File(train)
        labels = np.array(f["labels"])
        new_train_db = train.replace("train", "infer-train")
        new_val_db = train.replace("train", "infer-val")
        
        val_rgbd = []
        val_labels = []
        val_bus = []
        
        train_rgbd = []
        train_labels = []
        train_bus = []
        
        # Fetch images from labels in the train set
        for idx in range(len(ds_labels)):
            rgbd = np.concatenate([ds_images[idx],ds_depths[idx]], axis=1)
            if ds_labels[idx] in labels:
                train_rgbd.apppend(rgbd)
                train_labels.append(ds_labels[idx])
                train_bus.append(ds_bus[idx])
            else:
                val_rgbd.apppend(rgbd)
                val_labels.append(ds_labels[idx])
                val_bus.append(ds_bus[idx])
                
        # Save
        train_db = h5py.File(new_train_db + ".new")
        train_db.create_dataset("rgbd", data=np.concatenate(train_rgbd))
        train_db.create_dataset("bus", data=np.array(train_bus))
        train_db.create_dataset("labels", data=np.array(train_labels))
        
        val_db = h5py.File(new_val_db + ".new")
        val_db.create_dataset("rgbd", data=np.concatenate(val_rgbd))
        val_db.create_dataset("bus", data=np.array(val_bus))
        val_db.create_dataset("labels", data=np.array(val_labels))
        
        train_db.close()
        val_db.close()
        
                          
                        
                          
    

if __name__ == "__main__":
    main()
