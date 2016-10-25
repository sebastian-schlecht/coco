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
        self._compiled = False

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
            logger.info("Compiling training function.")
            self.train_fn = theano.function(self.train_inputs, self.train_outputs, updates=self.updates)
        if self.val_reader:
            logger.info("Compiling validation function.")
            self.val_fn = theano.function(self.val_inputs, self.val_outputs)
        if self.test_reader:
            logger.info("Compiling test function.")
            self.test_fn = theano.function(self.test_inputs, self.test_outputs)

        if self.inference:
            logger.info("Compiling inference function.")
            self.inference_fn = theano.function(self.inference_inputs, self.inference_outputs)

        self._compiled = True

    def infer(self, inputs):
        return self.inference_fn(inputs)

    def fit(self, n_epochs):
        if not self._compiled:
            raise AssertionError("Models are not compiled. Call 'compile()' first.")
        if self.train_reader:
            self.train_reader.start_daemons()
        if self.val_reader:
            self.val_reader.start_daemons()
        if self.test_reader:
            self.test_reader.start_daemons()

        batch_times = 0
        batch_index = 0

        # Make sure we know the lr to begin with
        assert 0 in self.lr_schedule

        for epoch in range(n_epochs):
            epoch_start = time.time()
            # Adapt LR if necessary
            if epoch in self.lr_schedule:
                val = self.lr_schedule[epoch]
                logger.info("Setting LR to " + str(val))
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
                    total_dur = n_epochs * (time_per_batch * self.train_reader.num_batches())
                    eta = time.time() + total_dur
                    localtime = time.asctime(time.localtime(eta))
                    logger.info("Average time per forward/backward pass: " + str(time_per_batch))
                    logger.info("Expected duration for training: %s" + str(total_dur) + "s")
                    logger.info("ETA: %s", str(localtime))
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
            epoch_end = time.time()
            logger.info("Epoch %i took %f seconds." % (epoch + 1, epoch_end - epoch_start))
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
    def __init__(self, inputs):
        if type(inputs) != list:
            raise AssertionError("Please provide a list of tensor variables as inputs.")
        self.output_layers = []
        self.input_layers = []
        self.last_layer = None
        self.registry = {}
        self.inputs = inputs
        # In case some sub-class implements this, call it
        self.output_layers = self.init()
        if len(self.output_layers) != 1:
            raise AssertionError("Only one output layer supported at the moment.")
        # Post init setup
        self._post_init()

    def init(self):
        """
        Override this method if you want to generate any network architecture from scratch
        :return:
        """
        raise NotImplementedError()

    def _post_init(self):
        for layer in self.output_layers:
            # Count parameters for outlet
            logger.debug(
                "Number of parameters for output '%s': %i" % (layer.name, lasagne.layers.count_params(layer, trainable=True)))

    def save(self, filename):
        network = self.output_layers[0]
        np.savez(filename, *lasagne.layers.get_all_param_values(network))

    def load(self, filename):
        # load network weights from model file
        network = self.output_layers[0]
        with np.load(filename) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
