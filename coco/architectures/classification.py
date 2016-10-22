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


class Resnet(Network):
    def __init__(self, input, num_classes, filter_config=(3, 4, 6, 3)):
        self.num_classes = num_classes
        self.filter_config = filter_config
        self.rc = 0
        super(Resnet, self).__init__(input=input)

    def init(self):
        def residual_block(l, increase_dim=False, projection=False):
            input_num_filters = l.output_shape[1]
            if increase_dim:
                first_stride = (2, 2)
                out_num_filters = input_num_filters * 2
            else:
                first_stride = (1, 1)
                out_num_filters = input_num_filters

            stack_1 = self.add("bn_%i_0" % self.rc, batch_norm(
                self.add("conv_%i_0" % self.rc,
                         ConvLayer(l, num_filters=out_num_filters, filter_size=(3, 3), stride=first_stride,
                                   nonlinearity=rectify,
                                   pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))
            stack_2 = self.add("bn_%i_1" % self.rc, batch_norm(
                self.add("conv_%i_1" % self.rc,
                         ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3, 3), stride=(1, 1),
                                   nonlinearity=None,
                                   pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))

            # add shortcut connections
            if increase_dim:
                if projection:
                    # projection shortcut, as option B in paper
                    projection = self.add("bn_%i_2" % self.rc, batch_norm(
                        self.add("conv_%i_2" % self.rc,
                                 ConvLayer(l, num_filters=out_num_filters, filter_size=(1, 1), stride=(2, 2),
                                           nonlinearity=None,
                                           pad='same', b=None, flip_filters=False))))
                    block = self.add("nonlinearity_%i" % self.rc, NonlinearityLayer(
                        self.add("elemwise_%i" % self.rc, ElemwiseSumLayer([stack_2, projection])),
                        nonlinearity=rectify))
                else:
                    # identity shortcut, as option A in paper
                    identity = self.add("exp_%i" % self.rc, ExpressionLayer(l, lambda X: X[:, :, ::2, ::2],
                                                                            lambda s: (
                                                                                s[0], s[1], s[2] // 2, s[3] // 2)))
                    padding = self.add("padding_%i" % self.rc,
                                       PadLayer(identity, [out_num_filters // 4, 0, 0], batch_ndim=1))
                    block = self.add("nonlinearity_%i" % self.rc, NonlinearityLayer(
                        self.add("elemwise_%i" % self.rc, ElemwiseSumLayer([stack_2, padding])), nonlinearity=rectify))
            else:
                block = self.add("nonlinearity_%i" % self.rc, NonlinearityLayer(
                    self.add("elemwise_%i" % self.rc, ElemwiseSumLayer([stack_2, l])), nonlinearity=rectify))

            self.rc += 1
            return block

        # Building the network
        l_in = self.add("input", InputLayer(shape=(None, 3, 224, 224), input_var=self.input))

        # First batch normalized layer and pool
        l = self.add("bn_0", batch_norm(
            self.add("conv_0", ConvLayer(l_in, num_filters=64, filter_size=(7, 7), stride=(2, 2), nonlinearity=rectify,
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))))
        l = self.add("pool_0", PoolLayer(l, pool_size=(3, 3), stride=(2, 2)))

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
        l = self.add("global_pool", GlobalPoolLayer(l))

        # fully connected layer
        self.add("fc", DenseLayer(
            l, num_units=self.num_classes,
            W=lasagne.init.HeNormal(),
            nonlinearity=softmax))
