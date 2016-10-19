import numpy as np
import theano.tensor as T

from lasagne.utils import floatX
from lasagne.layers import get_output
from lasagne.objectives import binary_crossentropy
from lasagne.objectives import categorical_crossentropy


def grad_scale(layer, scale):
    for param in layer.get_params(trainable=True):
        param.tag.grad_scale = floatX(scale)
    return layer


def compute_saliency(input, output, X, loss_function=binary_crossentropy, aggregate=lambda x: x):
    output = get_output(output, deterministic=True)
    scores = output.eval({input: X})

    pred = scores.argmax(axis=1)
    # Use some epsilon value to prevent rounding of the maximally likely classes to 100% and thus yield 0 error
    e = 0.0000001
    score = -loss_function(output[:, pred] - e, np.array([1])).sum()

    return aggregate(np.abs(T.grad(score, input).eval({input: X}))[0].transpose(1, 2, 0).squeeze())
