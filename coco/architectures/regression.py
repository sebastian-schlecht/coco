import theano.tensor as T

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

from coco.nn import Network, Scaffolder
from coco.losses import mse


class BURegressor(Network):
    def __init__(self, inputs, filter_config=(3, 4, 6, 3)):
        self.filter_config = filter_config
        self._input_layer = None
        self._output_layer = None
        super(BURegressor, self).__init__(inputs=inputs)

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
                          pad='same', W=lasagne.init.HeNormal(), flip_filters=False))

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
        self._input_layer = InputLayer(shape=(None, 3, 228, 304), input_var=self.inputs[0])
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
        l = DenseLayer(l, num_units=100, W=lasagne.init.HeNormal(), nonlinearity=None)
        
        # Output regressor unit
        self._output_layer = DenseLayer(
            l, num_units=1,
            W=lasagne.init.HeNormal(),
            nonlinearity=rectify)

        return [self._output_layer]


class BURegressionScaffolder(Scaffolder):
    def setup(self):
        input = T.tensor4("input")
        targets = T.fvector("targets")
        
        # We need to expand the dimensions since the net will produce matrices of shape (n, 1) 
        targets_reshaped = targets.dimshuffle((0, "x"))

        self.network = self.network_type([input], **self.args)
        output_layer = self.network.output_layers[0]

        prediction = lasagne.layers.get_output(output_layer)
        val_test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)

        train_loss = mse(prediction, targets_reshaped)
        val_test_loss = mse(val_test_prediction, targets_reshaped)

        # Weight decay
        all_layers = lasagne.layers.get_all_layers(output_layer)
        l2_penalty = lasagne.regularization.regularize_layer_params(
            all_layers, lasagne.regularization.l2) * 0.0001
        cost = train_loss + l2_penalty

        params = lasagne.layers.get_all_params(output_layer, trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            cost, params, learning_rate=self.lr, momentum=self.momentum)

        # Set proper variables
        self.train_inputs = [input, targets]
        self.val_inputs = [input, targets]
        self.test_inputs = [input, targets]
        self.inference_inputs = [input]

        self.train_outputs = [train_loss]
        self.val_outputs = [val_test_loss]
        self.test_outputs = [val_test_loss]

        self.inference_outputs = [val_test_prediction]
        
        self.lr_schedule = {
            1:  0.001,
            2:  0.01,
            10: 0.005,
            30: 0.002,
            35: 0.0001,
        }
