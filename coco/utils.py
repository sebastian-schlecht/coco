import numpy as np
import theano.tensor as T

from lasagne.utils import floatX
from lasagne.layers import get_output
from lasagne.objectives import binary_crossentropy


def grad_scale(layer, scale):
    """
    Scale individual layer gradients
    From https://github.com/dnouri/nolearn
    :param layer:
    :param scale:
    :return:
    """
    for param in layer.get_params(trainable=True):
        param.tag.grad_scale = floatX(scale)
    return layer


def compute_saliency(input_layer, output_layer, X, loss_function=binary_crossentropy, aggregate=lambda x: x):
    """
    Compute a static saliency representation for a given input tensor X
    From https://github.com/dnouri/nolearn
    :param input_layer:
    :param output_layer:
    :param X:
    :param loss_function:
    :param aggregate:
    :return:
    """
    output = get_output(output_layer, deterministic=True)
    scores = output.eval({input_layer.input_var: X})

    pred = scores.argmax(axis=1)
    # Use some epsilon value to prevent rounding of the maximally likely classes to 100% and thus yield 0 error
    e = 0.0000001
    score = -loss_function(output[:, pred] - e, np.array([1])).sum()

    return aggregate(np.abs(T.grad(score, input).eval({input: X}))[0].transpose(1, 2, 0).squeeze())
