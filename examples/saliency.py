import sys

import numpy as np

import theano
import theano.tensor as T

from PIL import Image

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

sys.path.append("../")

from coco.utils import compute_saliency


def build_cnn(input_var=None, n=5):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(
            ConvLayer(l, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride, nonlinearity=rectify,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]), nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad='same',
                             W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1, n):
        l = residual_block(l)

    # average pooling
    l = GlobalPoolLayer(l)

    # fully connected layer
    network = DenseLayer(
        l, num_units=10,
        W=lasagne.init.HeNormal(),
        nonlinearity=softmax)

    return network


def pixel_mean():
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    xs = []
    ys = []
    for j in range(5):
        d = unpickle('data/cifar-10-batches-py/data_batch_' + `j + 1`)
        x = d['data']
        y = d['labels']
        xs.append(x)
        ys.append(y)

    d = unpickle('data/cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs) / np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000], axis=0)
    return pixel_mean


def main():
    model = "./data/cifar_model_n5.npz"
    image = "./data/ship.png"

    n = 5

    # Open the image file
    img = np.array(Image.open(image)).transpose((2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    mean = np.load("./data/cifar_model_mean.npy")

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # load network weights from model file
    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    """
    Implementation according to
    https://arxiv.org/pdf/1312.6034.pdf
    """

    images = [
        {
            "image": img.copy()[:, :, :, ::-1],
            "transform": lambda x: x[:, ::-1]
        },
        {
            "image": img.copy(),
            "transform": lambda x: x
        }
    ]
    saliencies = []
    for map in images:
        current_img = map["image"]
        current_img /= 255.
        current_img -= mean
        tmp_saliency = compute_saliency(input_var, network, current_img)
        # Given the saliency map accross all input channels, we simply take the maximum value for each channel
        # in order to obtain spatial information only
        tmp_saliency = tmp_saliency.max(axis=2)
        tmp_saliency = map["transform"](tmp_saliency)
        saliencies.append(tmp_saliency)

    # Average them out
    saliency = np.array(saliencies).mean(axis=0)

    # According to
    upper = np.percentile(saliency, 95)
    lower = np.percentile(saliency, 30)

    mask = np.zeros(saliency.shape, dtype=np.float32)
    mask[saliency > upper] = 3.
    mask[saliency < lower] = 2.
    np.save("./data/saliency.npy", mask.astype(np.uint8))

    # The saved saliency can be used as initial mask for graphcut models
    print "Done"


if __name__ == "__main__":
    main()
