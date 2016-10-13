from lasagne.layers import ReshapeLayer, Conv2DLayer, ExpressionLayer


def ThinConvolution(input, n_filters=64, filter_size=(3, 3)):
    input_shape = input.output_shape

    transform = ReshapeLayer(input, (-1, 1, input_shape[2], input_shape[3]))

    conv = Conv2DLayer(transform, num_filters=n_filters, filter_size=filter_size)
    conv_shape = conv.output_shape

    reshape = ReshapeLayer(conv, (-1, input_shape[1], n_filters, conv_shape[2], conv_shape[3]))

    def accumulate(tensor):
        return tensor.mean(axis=1)

    return ExpressionLayer(reshape, accumulate, output_shape=(None, n_filters, conv_shape[2], conv_shape[3]))
