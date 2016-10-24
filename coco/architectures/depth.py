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

from coco.nn import Network


class ResidualDepth(Network):

    def __init__(self, input):
        super(ResidualDepth, self).__init__(input)

    def _residual_block_up(self, l, decrease_dim=False, projection=True, padding="same", conv_filter=(5, 5),
                           proj_filter=(5, 5), lr=1):
        input_num_filters = l.output_shape[1]
        if decrease_dim:
            out_num_filters = input_num_filters / 2
            l = self.add("upsample", Upscale2DLayer(l, 2))
        else:
            out_num_filters = input_num_filters

        stack_1 = self.add("expansive_bn_0", batch_norm(
            self.add("expansive_conv_0", ConvLayer(l, num_filters=out_num_filters, filter_size=conv_filter, stride=(1, 1), nonlinearity=rectify,
                                                   pad=padding, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False), lr)))
        stack_2 = self.add("expansive_bn_1", batch_norm(
            self.add("expansive_conv_1", ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1), nonlinearity=None,
                                                   pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False), lr)))

        if decrease_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    self.add("expansive_conv_proj", ConvLayer(l, num_filters=out_num_filters, filter_size=proj_filter, stride=(1, 1), nonlinearity=None,
                                                              pad=padding, b=None, flip_filters=False)))
                block = self.add("expansive_nonlin", NonlinearityLayer(self.add("expansive_elemwise", ElemwiseSumLayer(
                    [stack_2, projection])), nonlinearity=rectify))
            else:
                raise NotImplementedError()
        else:
            block = self.add("expansive_nonlin", NonlinearityLayer(self.add("expansive_elemwise", ElemwiseSumLayer(
                [stack_2, l])), nonlinearity=rectify))
        return block

    def _residual_block_down(self, l, increase_dim=False, projection=False, pad=True, force_output=None, lr=1):
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
        stack_1 = self.add("contractive_bn_0", batch_norm(
            self.add("contractive_conv_0", ConvLayer(l, num_filters=bottleneck, filter_size=(1, 1), stride=first_stride, nonlinearity=rectify,
                                                     pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False), lr)))
        stack_2 = self.add("contractive_bn_1", batch_norm(
            self.add("contractive_conv_1", ConvLayer(stack_1, num_filters=bottleneck, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify,
                                                     pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False), lr)))
        stack_3 = self.add("contractive_bn_2", batch_norm(
            self.add("contractive_conv_2", ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                                                     pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False), lr)))

        if increase_dim:
            if projection:
                projection = self.add("contractive_bn_proj", batch_norm(
                    self.add("contractive_conv_proj", ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2), nonlinearity=None,
                                                                pad='same', b=None, flip_filters=False), lr)))
                block = self.add("contractive_nonlin", NonlinearityLayer(self.add("contractive_elemwise", ElemwiseSumLayer(
                    [stack_3, projection])), nonlinearity=rectify))
            else:
                raise NotImplementedError()
        else:
            if projection:
                l = self.add("contractive_bn_proj", batch_norm(
                    self.add("contractive_conv_proj", ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                                                                pad='same', b=None, flip_filters=False), lr)))
            block = self.add("contractive_nonlin", NonlinearityLayer(self.add("contractive_elemwise", ElemwiseSumLayer(
                [stack_3, l])), nonlinearity=rectify))
        return block

    def _concat_compress(self, down, up, num_filters):
        # Recycle
        down = self._residual_block_down(down)
        # Concat
        l = self.add("concat", ConcatLayer([down, up]))
        # Compress channels
        l = self.add("compress_bn", batch_norm(self.add("compress_conv", ConvLayer(l, num_filters=num_filters, filter_size=(3, 3), stride=1, nonlinearity=rectify, pad='same',
                                                                                   W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))
        return l

    def init(self):
        l_in = self.add_input("input", InputLayer(
            shape=(None, 3, 228, 304), input_var=self.input))

        # First batch normalized layer
        l = self.add("bn_0", batch_norm(self.add("conv_0", ConvLayer(l_in, num_filters=64, filter_size=(7, 7), stride=(2, 2), nonlinearity=rectify, pad=3,
                                                                     W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))
        l = self.add("pool", PoolLayer(l, pool_size=(2, 2)))

        # Output is 64x60x80 at this point
        l = self._residual_block_down(l, projection=True, force_output=256)
        l = self._residual_block_down(l)

        l_2 = l

        # Output is 256x60x80 at this point
        l = self._residual_block_down(l, projection=True, increase_dim=True)
        for _ in range(1, 4):
            l = self._residual_block_down(l)
        l_3 = l

        # Output is 512x30x40 at this point

        l = self._residual_block_down(l, projection=True, increase_dim=True)
        for _ in range(1, 6):
            l = self._residual_block_down(l)
        l_4 = l

        # Output is 1024x16x20 at this point

        l = self._residual_block_down(l, projection=True, increase_dim=True)
        for _ in range(1, 3):
            l = self._residual_block_down(l)

        # Output is 2048x8x10 at this point
        l = self.add("compress_bn", batch_norm(self.add("compress_conv", ConvLayer(l, num_filters=1024, filter_size=(1, 1), stride=(1, 1), nonlinearity=None, pad="same",
                                                                                   W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))

        # first expansive block. seventh stack of residual,s output is
        # 512x16x20
        l = self._residual_block_up(l, decrease_dim=True, padding=1,
                                    conv_filter=(4, 4), proj_filter=(4, 4))
        l = self._residual_block_up(l)
        l_7 = l

        self._concat_compress(l_4, l_7, 512)

        # first expansive block. seventh stack of residuals, output is
        # 256x30x40
        l = self._residual_block_up(l, decrease_dim=True, padding=1,
                                    conv_filter=(4, 3), proj_filter=(4, 3))

        l = self._residual_block_down(l)
        l_8 = l

        self._concat_compress(l_3, l_8, 256)

        # residual block #8, output is 128x60x80
        l = self._residual_block_up(l, decrease_dim=True, padding=1,
                                    conv_filter=(4, 3), proj_filter=(4, 3))

        l = self._residual_block_down(l)
        l_9 = l

        self._concat_compress(l_2, l_9, 128)

        # residual block #9, output is 64x120x160
        l = self._residual_block_up(l, decrease_dim=True)
        l = self._residual_block_down(l)

        # Final convolution
        l = self.add_output("final_conv", ConvLayer(l, num_filters=1, filter_size=(3, 3), stride=(1, 1), nonlinearity=rectify, pad="same",
                                                    W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
