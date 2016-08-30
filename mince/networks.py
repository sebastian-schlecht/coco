import lasagne


class LeNet(object):
    @classmethod
    def build(cls, input_var, n_classes):
        # As a third model, we'll create a CNN of two convolution + pooling stages
        # and a fully-connected hidden layer in front of the output layer.

        # Input layer, as usual:
        network = lasagne.layers.InputLayer(shape=(None, 3, 224, 224),
                                            input_var=input_var)
        # This time we do not apply input dropout, as it tends to work less well
        # for convolutional layers.

        # Convolutional layer with 32 kernels of size 5x5. Strided and padded
        # convolutions are supported as well; see the docstring.
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        # Expert note: Lasagne provides alternative convolutional layers that
        # override Theano's choice of which implementation to use; for details
        # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

        # Max-pooling layer of factor 2 in both dimensions:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

        # And, finally, the output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=n_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

        return network
