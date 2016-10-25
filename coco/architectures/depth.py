import theano
import theano.tensor as T
import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer, Upscale2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import batch_norm
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.nonlinearities import rectify
from lasagne.layers import ConcatLayer

from coco.nn import Network, Scaffolder
from coco.losses import mse


class DepthPredictionScaffolder(Scaffolder):
    def setup(self):
        input = T.tensor4("input")
        targets = T.tensor3("targets")
        targets_reshaped = targets.dimshuffle((0, "x", 1, 2))

        self.network = self.network_type([input])
        output_layer = self.network.output_layers[0]

        prediction = lasagne.layers.get_output(output_layer)
        val_test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)

        # Compile different functions for the phases
        train_loss = mse(prediction, targets_reshaped, bounded=True, lower_bound=0.1, upper_bound=10.)
        val_test_loss = mse(val_test_prediction, targets_reshaped, bounded=True, lower_bound=0.1, upper_bound=10.)

        # Weight decay
        all_layers = lasagne.layers.get_all_layers(output_layer)
        l2_penalty = lasagne.regularization.regularize_layer_params(
            all_layers, lasagne.regularization.l2) * 0.0001
        cost = train_loss + l2_penalty

        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            cost, params, learning_rate=self.lr, momentum=0.9)

        # Set proper variables
        self.train_inputs = [input, targets]
        self.val_inputs = [input, targets]
        self.test_inputs = [input, targets]
        self.inference_inputs = [input]

        self.train_outputs = [train_loss]
        self.val_outputs = [val_test_loss]
        self.test_outputs = [val_test_loss]

        self.inference_outputs = [val_test_prediction]


class ResidualDepth(Network):
    def __init__(self, input, k=1):
        """
        Constructor
        :param input: Input expression
        :param k: Initial filter scaling factor
        :return:
        """
        self.k = k
        super(ResidualDepth, self).__init__(input)

    def _residual_block_up(self, l, decrease_dim=False, projection=True, padding="same", conv_filter=(3, 3),
                           proj_filter=(3, 3)):
        input_num_filters = l.output_shape[1]
        if decrease_dim:
            out_num_filters = input_num_filters / 2
            l = Upscale2DLayer(l, 2)
        else:
            out_num_filters = input_num_filters

        bottleneck = out_num_filters // 4
        stack_1 = batch_norm(
            ConvLayer(l, num_filters=bottleneck, filter_size=(1, 1), stride=(1, 1),
                      nonlinearity=rectify,
                      pad="same", W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(
            ConvLayer(stack_1, num_filters=bottleneck, filter_size=conv_filter, stride=(1, 1),
                      nonlinearity=rectify,
                      pad=padding, W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_3 = batch_norm(
            ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1),
                      nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        if decrease_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=proj_filter, stride=(1, 1),
                              nonlinearity=None,
                              pad=padding, b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_3, projection]), nonlinearity=rectify)
            else:
                raise NotImplementedError()
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]), nonlinearity=rectify)
        return block

    def _residual_block_down(self, l, increase_dim=False, projection=False, force_output=None):
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
            ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1),
                      nonlinearity=None,
                      pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

        if increase_dim:
            if projection:
                projection = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                              nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer(
                    [stack_3, projection]), nonlinearity=rectify)
            else:
                raise NotImplementedError()
        else:
            if projection:
                l = batch_norm(
                    ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1),
                              nonlinearity=None,
                              pad='same', b=None, flip_filters=False))
            block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]), nonlinearity=rectify)
        return block

    def _concat_compress(self, down, up, num_filters):
        # Recycle
        down = self._residual_block_down(down)
        # Concat
        l = ConcatLayer([down, up])
        # Compress channels
        l = batch_norm(
            ConvLayer(l, num_filters=num_filters, filter_size=(3, 3),
                      stride=1, nonlinearity=rectify, pad='same',
                      W=lasagne.init.HeNormal(gain='relu'),
                      flip_filters=False))
        return l

    def init(self):
        assert len(self.inputs) == 1

        l_in = InputLayer(
            shape=(None, 3, 228, 304), input_var=self.inputs[0])

        # First batch normalized layer
        l = batch_norm(
            ConvLayer(l_in, num_filters=64, filter_size=(7, 7), stride=(2, 2),
                      nonlinearity=rectify, pad=3,
                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        l = PoolLayer(l, pool_size=(2, 2))

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
        l = batch_norm(ConvLayer(l, num_filters=1024, filter_size=(1, 1),
                                 stride=(1, 1), nonlinearity=None, pad="same",
                                 W=lasagne.init.HeNormal(gain='relu'),
                                 flip_filters=False))

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
        l = ConvLayer(l, num_filters=1, filter_size=(1, 1), stride=(1, 1), nonlinearity=rectify,
                      pad="same",
                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)
        return [l]
