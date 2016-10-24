import sys

import numpy as np

import theano
import theano.tensor as T

from PIL import Image, ImageOps

import lasagne

sys.path.append("../")

from coco.architectures.classification import Resnet
from coco.utils import compute_saliency


def main():
    model = "./data/resnet50-food-101.npz"
    image = "./data/sushi.jpg"

    # Open the image file
    img = Image.open(image)
    img = np.array(img).transpose((2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    mean = np.load("./data/food-101-train.npy").astype(np.float32)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    resnet = Resnet(input_var, 101)
    name, network = resnet.output_layers.items()[0]
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # load network weights from model file
    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    """
    Implementation according to
    https://arxiv.org/pdf/1312.6034.pdf
    """

    images = [
        {
            "image": img.copy()[:, :, :, ::-1],
            "transform": lambda x: x[:, ::-1]
        },
        {
            "image": img.copy(),
            "transform": lambda x: x
        }
    ]
    saliencies = []
    for map in images:
        current_img = map["image"]
        current_img -= mean

        current_img = current_img[:, :, 16:240, 16:240]
        tmp_saliency = compute_saliency(input_var, network, current_img)
        # Given the saliency map accross all input channels, we simply take the maximum value for each channel
        # in order to obtain spatial information only
        tmp_saliency = tmp_saliency.max(axis=2)
        tmp_saliency = map["transform"](tmp_saliency)
        saliencies.append(tmp_saliency)

    # Average them out
    saliency = np.array(saliencies).mean(axis=0)

    # According to
    np.save("./data/saliency.npy", saliency)

    # The saved saliency can be used as initial mask for graphcut models
    print "Done"


if __name__ == "__main__":
    main()
