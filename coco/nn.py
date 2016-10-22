import collections
import joblib
import logging

from utils import grad_scale as param_grad_scale

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Network(object):
    def __init__(self, input):
        self._layers = collections.OrderedDict()
        self.last_layer = None
        self.input = input
        # In case some sub-class implements this, call it
        self.init()

    def init(self):
        """
        Override this method if you want to generate any network achitecture from scratch
        :return:
        """
        pass

    def add(self, name, layer, grad_scale=1):
        """
        Register a layer instance with this network. Eventually, also scale its gradients individually
        :param name:
        :param layer:
        :param lr:
        :return:
        """
        if name in self._layers:
            raise AssertionError("Layer names must be unique throughout the net. Layer %s already registered." % name)
        self._layers[name] = layer
        self.last_layer = layer

        # Apply individual learning rate to all trainable parameters
        if grad_scale != 1:
            param_grad_scale(layer, grad_scale)
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
