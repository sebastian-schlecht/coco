import logging
import time

import numpy as np
import theano

import lasagne

from job import Job

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Scaffolder(object):
    """
    Object to scaffold network training, validation, test and inference
    """
    PHASE_TRAIN = "phase_train"
    PHASE_VAL = "phase_val"
    PHASE_TEST = "phase_test"

    STATE_SETUP = "state_setup"
    STATE_RUN = "state_run"
    STATE_FINISH = "state_finish"

    def __init__(self, network_type, train_reader=None, val_reader=None, test_reader=None, inference=False, **kwargs):
        if not train_reader and not inference:
            raise AssertionError("Need either a training graph or an inference graph to compile.")

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
        self.lr_schedule = {1: 0.1}

        self.train_fn = None
        self.val_fn = None
        self.test_fn = None
        self.inference_fn = None

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.network_type = network_type
        self.args = kwargs
        self.network = None
        self.train_reader = train_reader
        self.val_reader = val_reader
        self.test_reader = test_reader
        self.inference = inference

        self.lr = theano.shared(lasagne.utils.floatX(0.01))
        self.momentum = theano.shared(lasagne.utils.floatX(0.9))

        self.setup()

    def setup(self):
        """
        Main routine to setup
        :return:
        """
        raise NotImplementedError()

    def compile(self):
        """
        Compile all expressions needed
        :return:
        """
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
        """
        Infer predictions from input values
        :param inputs:
        :return:
        """
        return self.inference_fn(inputs)

    def fit(self, n_epochs, job_name=None, snapshot=None, parallelism=1, lr_schedule=None, momentum=None):
        """
        Train the network
        :param n_epochs:
        :param job_name:
        :param snapshot:
        :param parallelism:
        :return:
        """
        if not self._compiled:
            raise AssertionError("Models are not compiled. Call 'compile()' first.")

        current_job = Job(job_name)
        current_job.set("state", Scaffolder.STATE_SETUP)
        if self.train_reader:
            self.train_reader.start_daemons(parallelism=parallelism)
        if self.val_reader:
            self.val_reader.start_daemons()
        if self.test_reader:
            self.test_reader.start_daemons()

        if not lr_schedule:
            lr_schedule = self.lr_schedule
        # Make sure we know the lr to begin with
        assert 1 in lr_schedule
        
        # Adjust momentum if applicable
        if momentum:
            self.momentum.set_value(lasagne.utils.floatX(momentum))

        current_job.set("state", Scaffolder.STATE_RUN)

        for epoch in range(1, n_epochs + 1):
            current_job.set("epoch", epoch)

            batch_times = 0
            batch_index = 0
            epoch_start = time.time()
            # Adapt LR if necessary
            if epoch in lr_schedule:
                val = lr_schedule[epoch]
                logger.info("Setting LR to " + str(val))
                self.lr.set_value(lasagne.utils.floatX(val))
            logger.info("Training Epoch %i" % (epoch))
            # Train for one epoch
            current_job.set("phase", Scaffolder.PHASE_TRAIN)
            for batch in self.train_reader.iterate():
                inputs, targets = batch
                batch_start = time.time()
                err = self.train_fn(inputs, targets)
                batch_end = time.time()
                batch_times += (batch_end - batch_start)
                batch_index += 1
                self.train_losses.append(err)
                self.on_batch_end(Scaffolder.PHASE_TRAIN)
                if batch_index % 100 == 0:
                    current_job.set("train_losses", self.train_losses)

                if batch_index == 20 and epoch == 1:
                    time_per_batch = batch_times / batch_index
                    total_dur = n_epochs * (time_per_batch * self.train_reader.num_batches())
                    eta = time.time() + total_dur
                    localtime = time.asctime(time.localtime(eta))
                    logger.info("Average time per forward/backward pass: " + str(time_per_batch))
                    logger.info("Expected duration for training: " + str(total_dur) + "s")
                    logger.info("ETA: %s", str(localtime))
            
            # In case we do have less than 100 batches, write back at the end to make sure we've got values
            current_job.set("train_losses", self.train_losses)

            if self.val_reader:
                current_job.set("phase", Scaffolder.PHASE_VAL)
                for batch in self.val_reader.iterate():
                    inputs, targets = batch
                    err = self.val_fn(inputs, targets)
                    self.val_losses.append(err)
                    self.on_batch_end(Scaffolder.PHASE_VAL)
                current_job.set("val_losses", self.val_losses)

            if self.test_reader:
                current_job.set("phase", Scaffolder.PHASE_TEST)
                for batch in self.test_reader.iterate():
                    inputs, targets = batch
                    err = self.test_fn(inputs, targets)
                    self.test_losses.append(err)
                    self.on_batch_end(Scaffolder.PHASE_TEST)
                current_job.set("test_losses", self.test_losses)

            epoch_end = time.time()
            logger.info("Epoch %i took %f seconds." % (epoch, epoch_end - epoch_start))
            if snapshot:
                logger.info("Saving model state to %s." % snapshot)
                self.save(snapshot)
            self.on_epoch_end()

        # Job done
        current_job.set("state", Scaffolder.STATE_RUN)

    def save(self, filename):
        """
        Store underlying network interface's weights
        :param filename:
        :return:
        """
        logger.info("Saving parameters to file '%s'" % filename)
        if self.network:
            self.network.save(filename)
        else:
            raise AssertionError("Network instance hasn't been create yet.")

    def load(self, filename, strict=True):
        """
        Load underlying network interface's weights from disk
        :param filename: Name of the parameter file
        :param loosely: Do not throw when shapes won't match. Just skip that layer instance
        :return:
        """
        logger.info("Loading parameters from file '%s'" % filename)
        if self.network:
            if not strict:
                self.network.load_non_strict(filename)
            else:
                self.network.load(filename)
        else:
            raise AssertionError("Network instance hasn't been create yet.")

    def on_epoch_end(self):
        """
        Epoch hook
        :return:
        """
        pass

    def on_batch_end(self, phase):
        """
        On batch end hook
        :param phase:
        :return:
        """
        pass


class Network(object):
    
    TRANSPLANT_MODE_DEFAULT = 1
    
    def __init__(self, inputs, **kwargs):
        
        if type(inputs) != list:
            raise AssertionError("Please provide a list of tensor variables as inputs.")
        
        self.output_layers = []
        self.input_layers = []
        self.last_layer = None
        self.registry = {}
        self.inputs = inputs
        
        # Weight transplant mode
        self.transplant_mode = Network.TRANSPLANT_MODE_DEFAULT
        if "transplant_mode" in kwargs:
            self.transplant_mode = kwargs["transplant_mode"]
        
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
        """
        Perform some post init sanity checks
        :return:
        """
        for layer in self.output_layers:
            # Count parameters for outlet
            logger.info(
                "Number of parameters for output '%s': %i" % (
                    layer.name, lasagne.layers.count_params(layer, trainable=True)))

    def save(self, filename):
        """
        Save network weights to disk
        :param filename:
        :return:
        """
        network = self.output_layers[0]
        np.savez(filename, *lasagne.layers.get_all_param_values(network))

    def load(self, filename):
        """
        Load network weights from disk
        :param filename:
        :return:
        """
        # load network weights from model file
        network = self.output_layers[0]
        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        
    def _transplant(self, variable, value):
        """
        Try to transplant values into the target variable even if the shapes won't match
        :param variable: Tensor variable
        :param value: The new weights
        """
        if self.transplant_mode == Network.TRANSPLANT_MODE_DEFAULT:
            # Try to transplant weights
            old_value = variable.get_value()
            # First thing we try is to simply assign the values
            if old_value.shape == value.shape:
                variable.set_value(value)
                return
            
            # If that didn't work we try to partially assign the weights to the filters 
            backup = variable.get_value()
            try:
                assert old_value.ndim == value.ndim
                if old_value.size > value.size:
                    logger.info("Parameter blob with shape %s needs transplant. Trying to fit source weights into target blob." % str(old_value.shape))
                    # We currently only support the case that we can align the old values nicely inside the new blob
                    if old_value.ndim == 4:
                        old_value[0:value.shape[0], 0:value.shape[1], 0:value.shape[2], 0:value.shape[3]] = value
                    elif old_value.ndim == 2:
                        old_value[0:value.shape[0], 0:value.shape[1]] = value
                else:
                    logger.info("Skipping blob with shape %s. Too many source parameters." % str(old_value.shape))
                     
            except Exception as e:
                logger.warn("Something went wrong while trying to transplant filters. Rolling back operation.")
                logger.warn("Original warning: %s " % str(e))
                variable.set_value(backup)
            
                
        else:
            raise ValueError("Unsupported transplant mode %s" % str(self.transplant_mode))
        
    def load_non_strict(self, filename):
        network = self.output_layers[0]
        with np.load(filename) as f:
            values = [f['arr_%d' % i] for i in range(len(f.files))]
            
        params = lasagne.layers.get_all_params(network)
        if len(params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                         (len(values), len(params)))

        for p, v in zip(params, values):
            if p.get_value().shape != v.shape:
                self._transplant(p, v)
            else:
                p.set_value(v)
