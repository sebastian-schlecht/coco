import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer, Upscale2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import batch_norm
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.nonlinearities import rectify
from lasagne.layers import ConcatLayer


def residual_unet(input_var=None, nu=1, connectivity=0):
    def residual_block_up(l, decrease_dim=False, projection=True, padding="same", conv_filter=(5, 5),
                          proj_filter=(5, 5)):
        input_num_filters = l.output_shape[1]

        if decrease_dim:
            out_num_filters = input_num_filters / 2
            l = Upscale2DLayer(l, 2)
        else:
            out_num_filters = input_num_filters
        stack_1 = batch_norm(
            ConvLayer(l, num_filters=out_num_filters, filter_size=conv_filter, stride=(1, 1), nonlinearity=rectify,
                      pad=padding, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        if decrease_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=proj_filter, stride=(1, 1), nonlinearity=None,
                              pad=padding, b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]), nonlinearity=rectify)
            else:
                raise NotImplementedError()
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]), nonlinearity=rectify)
        return block

    def residual_block(l, increase_dim=False, projection=False, pad=True, force_output=None):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        if force_output:
            out_num_filters = force_output

        bottleneck = out_num_filters // 4
        stack_1 = batch_norm(
            ConvLayer(l, num_filters=bottleneck, filter_size=(1, 1), stride=first_stride, nonlinearity=rectify,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            ConvLayer(stack_1, num_filters=bottleneck, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_3 = batch_norm(
            ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        if increase_dim:
            if projection:
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, projection]), nonlinearity=rectify)
            else:
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2] // 2, s[3] // 2))
                padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, padding]), nonlinearity=rectify)
        else:
            if projection:
                l = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
            block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]), nonlinearity=rectify)

        return block

    l_in = InputLayer(shape=(None, 3, 228, 304), input_var=input_var)

    # First batch normalized layer
    l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(7, 7), stride=(2, 2), nonlinearity=rectify, pad=3,
                             W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    l = PoolLayer(l, pool_size=(2, 2))

    # Output is 64x60x80 at this point
    l = residual_block(l, projection=True, force_output=256)
    l = residual_block(l)

    l_2 = l

    # Output is 256x60x80 at this point
    l = residual_block(l, projection=True, increase_dim=True)
    for _ in range(1, 4):
        l = residual_block(l)
    l_3 = l

    # Output is 512x30x40 at this point

    l = residual_block(l, projection=True, increase_dim=True)
    for _ in range(1, 6):
        l = residual_block(l)
    l_4 = l

    # Output is 1024x16x20 at this point

    l = residual_block(l, projection=True, increase_dim=True)
    for _ in range(1, 3):
        l = residual_block(l)

    # Output is 2048x8x10 at this point
    l = batch_norm(ConvLayer(l, num_filters=1024, filter_size=(1, 1), stride=(1, 1), nonlinearity=None, pad="same",
                             W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first expansive block. seventh stack of residual,s output is 512x16x20
    l = residual_block_up(l, decrease_dim=True, padding=1, conv_filter=(4, 4), proj_filter=(4, 4))
    for _ in range(1, nu):
        l = residual_block(l)
    l_7 = l
    if connectivity > 0:
        l = ConcatLayer([l_4, l_7])
        # Compress channels
        l = batch_norm(ConvLayer(l, num_filters=512, filter_size=(3, 3), stride=1, nonlinearity=rectify, pad='same',
                                 W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # first expansive block. seventh stack of residuals, output is 256x30x40
    l = residual_block_up(l, decrease_dim=True, padding=1, conv_filter=(4, 3), proj_filter=(4, 3))
    for _ in range(1, nu):
        l = residual_block(l)
    l_8 = l

    if connectivity > 1:
        l = ConcatLayer([l_3, l_8])
        l = batch_norm(ConvLayer(l, num_filters=256, filter_size=(3, 3), stride=1, nonlinearity=rectify, pad='same',
                                 W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # residual block #8, output is 128x60x80
    l = residual_block_up(l, decrease_dim=True, padding=1, conv_filter=(4, 3), proj_filter=(4, 3))
    for _ in range(1, nu):
        l = residual_block(l)
    l_9 = l

    if connectivity > 2:
        l = ConcatLayer([l_2, l_9])
        l = batch_norm(ConvLayer(l, num_filters=128, filter_size=(3, 3), stride=1, nonlinearity=rectify, pad='same',
                                 W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    # residual block #9, output is 64x120x160
    l = residual_block_up(l, decrease_dim=True)
    for _ in range(1, nu):
        l = residual_block(l)

    # Final convolution
    l = ConvLayer(l, num_filters=1, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad="same",
                  W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
    return l
