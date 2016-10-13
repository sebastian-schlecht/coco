import numpy as np
np.random.seed(42)
import os, sys, inspect, time
# This example requires theano & lasagne
import theano
import theano.tensor as T
import lasagne
import numpy as np

"""
Make the lib available here
"""
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from mince.database_builder import HDF5ClassDatabaseBuilder
from mince.database_reader import HDF5DatabaseReader
from mince.augmentations import rot_zoom_crop, mult_rgb
from mince.multiprocess import MultiProcessor
from mince.networks import resnet_50, lenet, resnet_34


"""
Preprocessor function
"""

mean = np.load("/data/data/food-101-train.npy")


def process(images, labels):
    w = 224
    h = 224
    images = images.astype(theano.config.floatX).copy()
    images_cropped = np.zeros((images.shape[0],images.shape[1], h,w),dtype=theano.config.floatX)
    
    for i in range(images.shape[0]):
        images[i] -= mean
        cy = 16
        cx = 16
        images_cropped[i] = images[i,:,cy:cy+h,cx:cx+w]

    return images_cropped, labels

def augment(images, labels):
    w = 224
    h = 224
    
    # Convert
    images = images.astype(theano.config.floatX).copy()
    images_cropped = np.zeros((images.shape[0],images.shape[1], h,w),dtype=theano.config.floatX)
    for i in range(images.shape[0]):
        p = np.random.randint(1)
        if p == 1:
            images[i] = images[i,:,:,::-1]
        
        # Rotate
        angle = np.random.randint(-5, 5)
        images[i] = rot_zoom_crop(images[i],angle, 1)
        
        # Sub mean    
        images[i] -= mean
        
        # Crop
        cy = np.random.randint(0, 32)
        cx = np.random.randint(0, 32)
        images_cropped[i] = images[i,:,cy:cy+h,cx:cx+w]
    
    return images_cropped, labels


"""
Main program
"""
if __name__ == "__main__":
    """
    Mince part
    """
    print "Building and reading database"

    # Target db location prefix
    db = '/data/data/food-101'

    # Folder holding subfolders, one for each class
    folder = '/nas/01_Datasets/Food/food-101/images'

    # Use helper to parse the folder
    classes = HDF5ClassDatabaseBuilder.parse_folder(folder)
    n_classes = len(classes)

    # Build a db from a set of images
    # In case force=false, we do not recreate the db if it's already there!
    train_db, val_db = HDF5ClassDatabaseBuilder.build(db, folder, shape=(256, 256), force=False)

    # Batch size to use during training
    batch_size = 64
    # Prepare the training reader for read access. This is necessary when combining it with multiprocessors
    train_reader = HDF5DatabaseReader()
    train_reader.setup_read(train_db, randomize_access=False)
    # Create a multiprocessor object which manages data loading and transformation daemons
    train_processor = MultiProcessor(train_reader, func=augment, batch_size=batch_size)
    # Start the daemons and tell them to use the databuilder we just setup to pull data from disk
    train_processor.start_daemons()

    # We also need to read validation data. It's way less so we just do in in the main thread
    # and don't start any daemons
    val_reader = HDF5DatabaseReader()
    val_reader.setup_read(val_db)
    val_processor = MultiProcessor(val_reader, batch_size=batch_size, func=process)

    """
    Lasagne part
    """
    print "Building and compiling network"
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')

    # Careful. We feed one-hot coded labels
    target_var = T.imatrix('targets')
    network = resnet_50(input_var, n_classes)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # add weight decay
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    loss = loss + l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = theano.shared(lasagne.utils.floatX(0.1))
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=lr, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

    """
    Training procedure
    """
    print "Starting training"

    n_epochs = 40
    
    bt = 0
    bidx = 0
    
    learning_schedule = {
        0: 0.005,
        1: 0.1,
        25: 0.01,
        35: 0.001
    }

    for epoch in range(n_epochs):
        print "Training Epoch %i" % (epoch + 1)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if epoch in learning_schedule:
            print "Setting LR to: ", learning_schedule[epoch]
            lr.set_value(learning_schedule[epoch])
        for batch in train_processor.iterate():
            bts = time.time()
            inputs, targets = batch
            err = train_fn(inputs, targets)
            train_err += err
            train_batches += 1
            
            bte = time.time()
            bt += (bte - bts)
            bidx += 1
            if bidx == 20 and epoch == 0:
                tpb = bt / bidx
                print "Average time per forward/backward pass: " + str(tpb)
                eta = time.time() + n_epochs * (tpb * (train_processor.num_samples()/batch_size))
                localtime = time.asctime( time.localtime(eta) )
                print "ETA: ", localtime

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in val_processor.iterate():
            inputs, targets = batch
            err, acc, pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        # dump the network weights to a file :
        np.savez('resnet50-food-101.npz', *lasagne.layers.get_all_param_values(network))
