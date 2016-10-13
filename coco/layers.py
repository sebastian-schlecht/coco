from lasagne.layers.conv import BaseConvLayer
from lasagne import init
from lasagne import nonlinearities

import theano.tensor as T
import numpy as np

__all__ = [
    "SparseConvolutionLayer"
]


class Thin2DConvolutionLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Thin2DConvolutionLayer, self).__init__(incoming, num_filters, filter_size,
                                                       stride, pad, untie_biases, W, b,
                                                       nonlinearity, flip_filters, n=2,
                                                       **kwargs)
        self.convolution = convolution

    def get_W_shape(self):
        """
        Provide a different shape for the weight matrix. Our filter has a depth of 1 as it operates on every input
        feature map individually
        :return:
        """
        num_input_channels = 1
        return (self.num_filters, num_input_channels) + self.filter_size

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad

        # In a first step, we align the input feature maps along their spatial dimension
        new_shape = (self.input_shape[0], 1, -1, self.input_shape[3])
        transformed = T.reshape(input, newshape=new_shape)

        # Convolve
        conved = self.convolution(transformed, self.W,
                                  new_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)


        immediate_conv_shape = self.get_output_shape_for(new_shape)

        # Transform back
        original_shape = self.output_shape
        if np.prod(immediate_conv_shape) != np.prod(original_shape):
            raise AssertionError("Shape mismatch! Cannot reshape from %s to %s" % (str(immediate_conv_shape), str(original_shape)))
        transformed_back = T.reshape(conved, original_shape)
        return transformed_back
