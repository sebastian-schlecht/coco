import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

from coco.nn import Network
from coco.utils import compute_saliency


class Resnet(Network):
    def __init__(self, inputs, num_classes, filter_config=(3, 4, 6, 3)):
        self.num_classes = num_classes
        self.filter_config = filter_config
        self._input_layer = None
        self._output_layer = None
        super(Resnet, self).__init__(inputs=inputs)

    def compute_saliency(self, x):
        return compute_saliency(self._input_layer, self._output_layer, x)

    def init(self):
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
                ConvLayer(l, num_filters=bottleneck, filter_size=(1, 1), stride=first_stride,
                          nonlinearity=rectify,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
            stack_2 = batch_norm(
                ConvLayer(stack_1, num_filters=bottleneck, filter_size=(3, 3), stride=(1, 1),
                          nonlinearity=rectify,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
            stack_3 = batch_norm(
                ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1),
                          nonlinearity=None,
                          pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

            # add shortcut connections
            if increase_dim:
                if projection:
                    # projection shortcut, as option B in paper
                    projection = batch_norm(
                        ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                  nonlinearity=None,
                                  pad='same', b=None, flip_filters=False))
                    block = NonlinearityLayer(
                        ElemwiseSumLayer([stack_3, projection]),
                        nonlinearity=rectify)
                else:
                    # identity shortcut, as option A in paper
                    identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2],
                                               lambda s: (
                                                   s[0], s[1], s[2] // 2, s[3] // 2))
                    padding = PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1)
                    block = NonlinearityLayer(
                        ElemwiseSumLayer([stack_3, padding]), nonlinearity=rectify)
            else:
                if projection:
                    l = batch_norm(
                        ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(1, 1), nonlinearity=None,
                                  pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(
                    ElemwiseSumLayer([stack_3, l]), nonlinearity=rectify)
            return block

        # Building the network
        self._input_layer = InputLayer(shape=(None, 3, 224, 224), input_var=self.inputs[0])
        self.input_layers.append(self._input_layer)

        # First batch normalized layer and pool
        l = batch_norm(
            ConvLayer(self._input_layer, num_filters=64, filter_size=(7, 7), stride=(2, 2), nonlinearity=rectify,
                      W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        l = PoolLayer(l, pool_size=(3, 3), stride=(2, 2))

        l = residual_block(l, force_output=256, projection=True)
        for _ in range(1, self.filter_config[0]):
            l = residual_block(l)

        l = residual_block(l, increase_dim=True, projection=True)
        for _ in range(1, self.filter_config[1]):
            l = residual_block(l)

        l = residual_block(l, increase_dim=True, projection=True)
        for _ in range(1, self.filter_config[2]):
            l = residual_block(l)

        l = residual_block(l, increase_dim=True, projection=True)
        for _ in range(1, self.filter_config[3]):
            l = residual_block(l)

        # average pooling
        l = GlobalPoolLayer(l)

        # fully connected layer
        self._output_layer = DenseLayer(
            l, num_units=self.num_classes,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax)

        return [self._output_layer]
