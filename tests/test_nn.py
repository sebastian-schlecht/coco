import unittest
import numpy as np

import theano.tensor as T
import lasagne

from coco.nn import Network


class TestNetwork(unittest.TestCase):
    def test_store_load(self):
        # We build a simple net and store it using lasagne and coco.
        # The parameter values should be the same for both cases
        def build_net():
            input_var = T.tensor4('inputs')
            n_classes = 10

            n = Network()

            l = n.add("input", lasagne.layers.InputLayer(shape=(None, 3, 224, 224),
                                                         input_var=input_var))
            l = n.add("conv_0", lasagne.layers.Conv2DLayer(
                l, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform()))
            l = n.add("pool_0", lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2)))
            l = n.add("conv_1", lasagne.layers.Conv2DLayer(
                l, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify))
            l = n.add("pool_1", lasagne.layers.MaxPool2DLayer(l, pool_size=(2, 2)))
            l = n.add("fc_0", lasagne.layers.DenseLayer(
                l,
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify))

            l = n.add("fc_1", lasagne.layers.DenseLayer(
                lasagne.layers.dropout(l, p=.5),
                num_units=n_classes,
                nonlinearity=lasagne.nonlinearities.softmax))

            return n, l

        # Build a net and store it twice, once using lasagne, once using coco
        coconet, lasagnenet = build_net()

        # Coco
        coconet.save('/tmp/coco_model.cocomodel')
        # Lasagne
        np.savez('/tmp/lasagne_model.npz', *lasagne.layers.get_all_param_values(lasagnenet))

        # Build a new coco net and load it from disk
        coconet, l = build_net()
        coconet.load('/tmp/coco_model.cocomodel')

        # Build a new lasagne net and load it from disk
        _, network = build_net()
        with np.load('/tmp/lasagne_model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

        # Compare
        params_a = lasagne.layers.get_all_param_values(l)
        params_b = lasagne.layers.get_all_param_values(network)
        for a, b in zip(params_a, params_b):
            self.assertTrue(np.allclose(a, b, rtol=1e-05, atol=1e-08))
