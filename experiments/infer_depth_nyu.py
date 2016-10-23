import sys
import os
import inspect

import theano
import theano.tensor as T
import numpy as np

import lasagne

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from coco.database_reader import HDF5DatabaseReader
from coco.multiprocess import MultiProcessor
from coco.architectures.depth import ResidualDepth
from coco.losses import mse
from coco.transformations import zoom_rotate, random_rgb, random_crop, normalize_images, downsample


def process_train(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)

    images, labels = zoom_rotate(images, labels)
    images, labels = random_rgb(images, labels)
    images, labels = random_crop(images, labels, size)

    images, labels = downsample(images, labels, (1, 2))

    return images, labels


def process_val(images, labels):
    images = images.astype(np.float32)
    labels = labels.astype(np.float32)

    assert images.shape[0] == labels.shape[0]

    size = (228, 304)

    images, labels = random_crop(images, labels, size, deterministic=True)
    images, labels = downsample(images, labels, (1, 2))

    return images, labels


def main():
    train_db = "/Users/sebastian/Desktop/f3d-train.h5"
    val_db = "/Users/sebastian/Desktop/f3d-val.h5"

    batch_size = 16

    train_reader = HDF5DatabaseReader(label_key="depths")
    train_reader.setup_read(train_db)

    val_reader = HDF5DatabaseReader()
    val_reader.setup_read(val_db)

    train_processor = MultiProcessor(
        train_reader, func=process_train, batch_size=batch_size)
    val_processor = MultiProcessor(
        val_reader, func=process_val, batch_size=batch_size)

    # Create net
    input = T.tensor4("input")
    targets = T.tensor3("targets")

    output_name, network = ResidualDepth(input).output_layers.items()[0]

    prediction = lasagne.layers.get_output(network)
    loss = mse(prediction, targets, upper_bound=10.)

    # add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(
        all_layers, lasagne.regularization.l2) * 0.0001
    loss = loss + l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = theano.shared(lasagne.utils.floatX(0.1))
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=lr, momentum=0.9)

    train_processor.start_daemons()

    train_fn = theano.function([input, targets], loss, updates=updates)

    epochs = 10

    for epoch in range(epochs):
        for batch in train_processor.iterate():
            inputs, targets = batch
            err = train_fn(inputs, targets)
            print err


if __name__ == "__main__":
    main()
