import collections
import joblib
import logging
import time

import numpy as np
import theano

import lasagne

from utils import grad_scale as param_grad_scale

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Scaffolder(object):
    """
    Object to scaffold network training, validation, test and inference
    """
    PHASE_TRAIN = 1
    PHASE_VAL = 2
    PHASE_TEST = 3

    def __init__(self, network_type, train_reader=None, val_reader=None, test_reader=None, inference=False):
        self.train_inputs = []
        self.val_inputs = []
        self.test_inputs = []
        self.inference_inputs = []

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.inference_outputs = []

        self.train_cost = None
        self.val_cost = None
        self.test_cost = None

        self.updates = None
        self.lr_schedule = {0: 0.1}

        self.train_fn = None
        self.val_fn = None
        self.test_fn = None
        self.inference_fn = None

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.network_type = network_type
        self.network = None
        self.train_reader = train_reader
        self.val_reader = val_reader
        self.test_reader = test_reader
        self.inference = inference

        self.lr = theano.shared(lasagne.utils.floatX(0.01))

        self.setup()

    def setup(self):
        """
        Main routine to setu
        :return:
        """
        raise NotImplementedError()

    def compile(self):
        if self.train_reader:
            logger.debug("Compiling training function.")
            self.train_fn = theano.function(self.train_inputs, self.train_outputs, updates=self.updates)
        if self.val_reader:
            logger.debug("Compiling validation function.")
            self.val_fn = theano.function(self.val_inputs, self.val_outputs)
        if self.test_reader:
            logger.debug("Compiling test function.")
            self.test_fn = theano.function(self.test_inputs, self.test_outputs)

        if self.inference:
            logger.debug("Compiling inference function.")
            self.inference_fn = theano.function(self.inference_inputs, self.inference_outputs)

    def infer(self, inputs):
        return self.inference_fn(inputs)

    def fit(self, n_epochs):
        batch_times = 0
        batch_index = 0

        # Make sure we know the lr to begin with
        assert 0 in self.lr_schedule

        for epoch in range(n_epochs):
            # Adapt LR if necessary
            if epoch in self.lr_schedule:
                val = self.lr_schedule[epoch]
                logger.debug("Setting LR to " + str(val))
                self.lr.set_value(lasagne.utils.floatX(val))
            logger.info("Training Epoch %i" % (epoch + 1))
            # Train for one epoch
            for batch in self.train_reader.iterate():
                inputs, targets = batch
                batch_start = time.time()
                err = self.train_fn(inputs, targets)
                batch_end = time.time()
                batch_times += (batch_end - batch_start)
                batch_index += 1
                if batch_index == 20 and epoch == 0:
                    time_per_batch = batch_times / batch_index
                    eta = time.time() + n_epochs * (time_per_batch * self.train_reader.num_samples())
                    localtime = time.asctime(time.localtime(eta))
                    logger.info("Average time per forward/backward pass: " + str(time_per_batch))
                    logger.info("ETA: ", localtime)
                self.train_losses.append(err)
                self.on_batch_end(Scaffolder.PHASE_TRAIN)

            if self.val_reader:
                for batch in self.val_reader.iterate():
                    inputs, targets = batch
                    err = self.val_fn(inputs, targets)
                    self.val_losses.append(err)
                    self.on_batch_end(Scaffolder.PHASE_VAL)

            if self.test_reader:
                for batch in self.test_reader.iterate():
                    inputs, targets = batch
                    err = self.test_fn(inputs, targets)
                    self.test_losses.append(err)
                    self.on_batch_end(Scaffolder.PHASE_TEST)
            self.on_epoch_end()

    def save(self, filename):
        if self.network:
            self.network.save(filename)
        else:
            raise AssertionError("Network instance hasn't been create yet.")

    def load(self, filename):
        if self.network:
            self.network.load(filename)
        else:
            raise AssertionError("Network instance hasn't been create yet.")

    def on_epoch_end(self):
        pass

    def on_batch_end(self, phase):
        pass


class Network(object):
    def __init__(self, input):
        self._layers = collections.OrderedDict()
        self.output_layers = collections.OrderedDict()
        self.input_layers = collections.OrderedDict()
        self.last_layer = None
        self.input = input
        # In case some sub-class implements this, call it
        self.init()
        for name, layer in self.output_layers.items():
            logger.debug(
                "Number of parameters for output '%s': %i" % (name, lasagne.layers.count_params(layer, trainable=True)))

    def init(self):
        """
        Override this method if you want to generate any network achitecture from scratch
        :return:
        """
        raise NotImplementedError()

    def add_output(self, name, layer, grad_scale=1):
        self.add(name, layer, grad_scale=grad_scale)
        if name in self.output_layers:
            raise AssertionError("Output layer names must be unique")
        self.output_layers[name] = layer
        return layer

    def add_input(self, name, layer):
        self.add(name, layer)
        if name in self.input_layers:
            raise AssertionError("Input layer names must be unique")
        self.input_layers[name] = layer
        return layer

    def add(self, name, layer, grad_scale=1):
        """
        Register a layer instance with this network. Eventually, also scale its gradients individually
        :param name:
        :param layer:
        :param lr:
        :return:
        """
        if name in self._layers:
            index = 0
            new_name = name
            while new_name in self._layers:
                new_name = name + ("_%i" % index)
                index += 1
            name = new_name
        self._layers[name] = layer
        self.last_layer = layer
        # Apply individual learning rate to all trainable parameters
        if grad_scale != 1:
            param_grad_scale(layer, grad_scale)

        shapes = [param.get_value().shape for param in layer.get_params(trainable=True)]
        counts = [np.prod(shape) for shape in shapes]
        logger.debug(
            "Layer '%s' output shape: %s with trainable parameters: %i" % (name, str(layer.output_shape), sum(counts)))
        return layer

    def save(self, filename, unwrap_shared=True, compress=2, **tags):
        data_store = {}
        for key, layer in self._layers.items():
            params = layer.get_params(unwrap_shared=unwrap_shared, **tags)
            values = [p.get_value() for p in params]
            data_store[key] = values
        joblib.dump(data_store, filename, compress=compress)

    def load(self, filename, **tags):
        data_store = joblib.load(filename)
        # If len doesn't match we don't care because we support partial initialization
        if len(data_store) != len(self._layers):
            logger.info("Found %i layers in data-store and %i layers registered in this net." % (
                len(data_store), len(self._layers)))
        for key in data_store:
            if key in self._layers:
                layer_params = self._layers[key].get_params(**tags)
                param_values = data_store[key]
                for p, v in zip(layer_params, param_values):
                    if p.get_value().shape != v.shape:
                        raise AssertionError(
                            "Cannot set parameter %s of layer %s because their shapes don't match: %s vs. %s" % (
                                str(p), str(key), str(p.get_value().shape), str(v.shape)))
                    else:
                        p.set_value(v)
