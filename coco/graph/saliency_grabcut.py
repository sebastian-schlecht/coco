from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from gmm import GMM
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import argparse
import os

try:
    import third_party.pymaxflow.pymaxflow as pymaxflow
except ImportError:
    import maxflow

import time
import sys

# Global constants
gamma = 50

# If energy changes less than CONVERGENCE_CRITERIA percent from the last iteration
# we will terminate
CONVERGENCE_CRITERON = 0.02

################################################################################
############################## TIMING CONSTRUCTS ###############################
################################################################################
start_time = 0


def tic():
    global start_time
    start_time = time.time()


def toc(task_label):
    global start_time
    print "%s took %0.4f s" % (task_label, time.time() - start_time)


################################################################################
############################ GENERAL I/O FUNCTIONS #############################
################################################################################
# get_args function
# Intializes the arguments parser and reads in the arguments from the command
# line.
#
# Returns: args dict with all the arguments
def get_args():
    parser = argparse.ArgumentParser(
        description='Implementation of the GrabCut algorithm.')
    parser.add_argument('-i', '--image-file', dest="image_file",
                        required=True,
                        help='Input image name along with its relative path')
    parser.add_argument('-s', '--saliency-file', dest="saliency_file",
                        required=True,
                        help='Input saliency image name along with its relative path')
    parser.add_argument('-n', '--num-iterations', dest="num_iterations",
                        type=int, default=10,
                        help='Number of iterations to run GrabCut for')
    parser.add_argument('-c', '--num-components', dest="num_components",
                        type=int, default=5,
                        help='Number of components in each GMM')
    parser.add_argument('-e', '--enable-user-interaction', dest="user_interaction",
                        action="store_true", default=False,
                        help='Flag to enable user interaction to provide more feedback')

    return parser.parse_args()


# load_image function
# Loads an image using matplotlib's built in image reader
# Note: Requires PIL (python imaging library) to be installed if the image is
# not a png
#
# Returns: img matrix with the contents of the image
def load_image(img_name):
    print 'Reading %s...' % img_name
    return plt.imread(img_name)


################################################################################
################################ UI FUNCTIONS  #################################
################################################################################
# RectSelector class
# Enables prompting user to select a rectangular area on a given image
class RectSelector:
    def __init__(self, ax):
        self.button_pressed = False
        self.start_x = 0
        self.start_y = 0
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.rectangle = []

    # Handles the case when the mouse button is initially pressed
    def on_press(self, event):
        self.button_pressed = True

        # Save the initial coordinates
        self.start_x = event.xdata
        self.start_y = event.ydata
        selected_rectangle = Rectangle((self.start_x, self.start_y),
                                       width=0, height=0, fill=False, linestyle='dashed')

        # Add new rectangle onto the canvas
        self.ax.add_patch(selected_rectangle)
        self.canvas.draw()

    # Handles the case when the mouse button is released
    def on_release(self, event):
        self.button_pressed = False

        # Check if release happened because of mouse moving out of bounds,
        # in which case we consider it to be an invalid selection
        if event.xdata == None or event.ydata == None:
            return
        x = event.xdata
        y = event.ydata

        width = x - self.start_x
        height = y - self.start_y
        selected_rectangle = Rectangle((self.start_x, self.start_y),
                                       width, height, fill=False, linestyle='solid')

        # Remove old rectangle and add new one
        self.ax.patches = []
        self.ax.add_patch(selected_rectangle)
        self.canvas.draw()
        xs = sorted([self.start_x, x])
        ys = sorted([self.start_y, y])
        self.rectangle = [xs[0], ys[0], xs[1], ys[1]]

        # Unblock plt
        plt.close()

    def on_move(self, event):
        # Check if the mouse moved out of bounds,
        # in which case we do not care about its position
        if event.xdata == None or event.ydata == None:
            return

        # If the mouse button is pressed, we need to update current rectangular
        # selection
        if self.button_pressed:
            x = event.xdata
            y = event.ydata

            width = x - self.start_x
            height = y - self.start_y

            selected_rectangle = Rectangle((self.start_x, self.start_y),
                                           width, height, fill=False, linestyle='dashed')

            # Remove old rectangle and add new one
            self.ax.patches = []
            self.ax.add_patch(selected_rectangle)
            self.canvas.draw()


# PolyLineSelector class
# Enables prompting user to select a series of points to indicate background
# in a semi-segmented image
class PolylineSelector:
    def __init__(self, ax, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.button_pressed = False
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.points = []

    def on_move(self, event):
        # pget the x and y pixel coords
        if self.button_pressed == True:
            x, y = event.xdata, event.ydata

            if event.inaxes:
                ax = event.inaxes  # the axes instance
                self.points.append((x, y))

                selected_circle = Circle((x, y), radius=3)

                self.ax.add_patch(selected_circle)
                self.canvas.draw()

    def on_press(self, event):
        self.button_pressed = True
        # get the x and y coords, flip y from top to bottom
        x, y = event.xdata, event.ydata
        # y = self.image_width - y
        if event.button == 1:
            if event.inaxes is not None:
                self.points.append((x, y))

    def on_release(self, event):
        self.button_pressed = False


# get_user_polyline
# Returns points drawn by a user on the image given as argument
def get_user_polyline(img):
    if img.shape[2] != 3:
        print 'This image does not have all the RGB channels, you do not need to work on it.'
        return

    # Initialize rectangular selector
    fig, ax = plt.subplots()
    selector = PolylineSelector(ax, img)

    ax.imshow(img)
    plt.show()

    return selector.points


# get_user_selection
# Returns coordinates of the bounding box the user draws on the given image
def get_user_selection(img):
    if img.shape[2] != 3:
        print 'This image does not have all the RGB channels, you do not need to work on it.'
        return

    # Initialize rectangular selector
    fig, ax = plt.subplots()
    selector = RectSelector(ax)

    # Show the image on the screen
    ax.imshow(img)
    plt.show()

    # Control reaches here once the user has selected a rectangle,
    # since plt.show() blocks.
    # Return the selected rectangle
    return selector.rectangle


################################################################################
########################## GRABCUT HELPER FUNCTIONS  ###########################
################################################################################
# Given an image and bounding box, initializes a foreground and a background
# GMM. The number of components can optionally be passed in.
def initialization(img, saliency, num_components=5, debug=True):

    height, width, _ = img.shape
    alpha = np.zeros((height, width), dtype=np.int8)

    upper = np.percentile(saliency, 95)
    lower = np.percentile(saliency, 30)

    assert img.shape[:2] == saliency.shape[:2]

    alpha[2:-2, 2:-2] = 1

    foreground_gmm = GMM(num_components)
    background_gmm = GMM(num_components)

    fg_clusters = foreground_gmm.initialize_gmm(img[saliency > upper])
    bg_clusters = background_gmm.initialize_gmm(img[saliency < lower])

    if debug:
        k = np.ones(alpha.shape, dtype=int) * -1
        k[saliency > upper] = fg_clusters[:]
        k[saliency < lower] = bg_clusters[:]
        visualize_clusters(img.shape, k, alpha, 0, "clusters")

        plt.imshow(alpha * 265)
        plt.show()
        for i in xrange(alpha.shape[0]):
            for j in xrange(alpha.shape[1]):
                print alpha[i, j],
            print ''

    return alpha, foreground_gmm, background_gmm


# Given an image, computes the total number of nodes and edges required, and
# returns a constructed graph object
def create_graph(img):
    num_neighbors = 8

    num_nodes = img.shape[0] * img.shape[1] + 2
    num_edges = img.shape[0] * img.shape[1] * num_neighbors

    g = maxflow.Graph[float](num_nodes, num_edges)

    # Creating nodes
    g.add_nodes(num_nodes - 2)

    return g


# Given a gmm and a list of pixels, computes the -log(prob) of each pixel belonging
# to the given GMM. This method does not consider which component was assigned
# to the pixel
#
# Currently unused in the implementation.
def get_total_unary_energy_vectorized(gmm, pixels, debug=False):
    # print k
    prob = 0.0
    for COMP in xrange(5):
        k = np.ones((pixels.shape[0],), dtype=int) * COMP
        pi_base = gmm.weights

        dets_base = np.array([gmm.gaussians[i].sigma_det for i in xrange(len(gmm.gaussians))])
        dets = dets_base[k]

        means_base = np.array([gmm.gaussians[i].mean for i in xrange(len(gmm.gaussians))])
        means = means_base[k]

        cov_base = np.array([gmm.gaussians[i].sigma_inv for i in xrange(len(gmm.gaussians))])
        cov = cov_base[k]

        term = pixels - means

        middle_matrix = np.array([np.sum(np.multiply(term, cov[:, :, 0]), axis=1),
                                  np.sum(np.multiply(term, cov[:, :, 1]), axis=1),
                                  np.sum(np.multiply(term, cov[:, :, 2]), axis=1)]).T

        log_prob = np.sum(np.multiply(middle_matrix, term), axis=1)

        if debug:
            print pi.shape
            print dets.shape
            print log_prob.shape

        prob += pi_base[COMP] * np.divide(np.exp(-0.5 * log_prob), ((2 * np.pi) ** 3) * dets)
    return -np.log(prob)


# Given a list of pixels and a gmm (gmms contains both the foreground and the
# background GMM, but alpha helps us pick the correct one), returns the -log(prob)
# of each belonging to the component specified by k.
#
# alpha - integer specifying background or foreground
# k - array with each element corresponding to which component the
#   corresponding pixel in the pixels array belongs to
# pixels - array of pixels
def get_unary_energy_vectorized(alpha, k, gmms, pixels, debug=False):
    pi_base = gmms[alpha].weights
    pi = pi_base[k].reshape(pixels.shape[0])
    pi[pi == 0] = 1e-15

    dets_base = np.array([gmms[alpha].gaussians[i].sigma_det for i in xrange(len(gmms[alpha].gaussians))])
    dets = dets_base[k].reshape(pixels.shape[0])
    dets[dets == 0] = 1e-15

    means_base = np.array([gmms[alpha].gaussians[i].mean for i in xrange(len(gmms[alpha].gaussians))])
    means = np.swapaxes(means_base[k], 1, 2)
    means = means.reshape((means.shape[0:2]))

    cov_base = np.array([gmms[alpha].gaussians[i].sigma_inv for i in xrange(len(gmms[alpha].gaussians))])
    cov = np.swapaxes(cov_base[k], 1, 3)
    cov = cov.reshape((cov.shape[0:3]))

    term = pixels - means
    middle_matrix = np.array([np.sum(np.multiply(term, cov[:, :, 0]), axis=1),
                              np.sum(np.multiply(term, cov[:, :, 1]), axis=1),
                              np.sum(np.multiply(term, cov[:, :, 2]), axis=1)]).T

    # Not really the log_prob, but a part of it
    log_prob = np.sum(np.multiply(middle_matrix, term), axis=1)

    if debug:
        print pi.shape
        print dets.shape
        print log_prob.shape

    return -np.log(pi) \
           + 0.5 * np.log(dets) \
           + 0.5 * log_prob


# Given an image (z), computes the expected difference between neighboring
# pixels, and returns the corresponding beta value.
def compute_beta_vectorized(z, debug=False):
    accumulator = 0
    m = z.shape[0]
    n = z.shape[1]

    vert_shifted = z - np.roll(z, 1, axis=0)
    temp = np.sum(np.multiply(vert_shifted, vert_shifted), axis=2)
    accumulator = np.sum(temp[1:, :])

    horiz_shifted = z - np.roll(z, 1, axis=1)
    temp = np.sum(np.multiply(horiz_shifted, horiz_shifted), axis=2)
    accumulator += np.sum(temp[:, 1:])

    num_comparisons = float(2 * (m * n) - m - n)
    if debug:
        print accumulator
        print num_comparisons
    beta = 1.0 / (2 * (accumulator / num_comparisons))

    return beta


# Given an image, and an optional neighborhood parameter, computes all the
# pairwise weights between neigboring pixels
#
# z - matrix of image pixels
# neighborhood - 'eight' for 8 neighborhood, 'four' for 4 neighborhood
# compute_dict - Computes a dictionary from 'pixel' to another dict that maps
#   'pixels' to pairwise energy. Hence we will have elements like:
#   dict[(5,6)] = dict[(4,6) -> 1.2, (6,6) -> 0.6, ...]. Only used to compare
#   to unvectorized version of compute_smoothness
def compute_smoothness_vectorized(z, neighborhood='eight', compute_dict=False, debug=False):
    FOUR_NEIGHBORHOOD = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    EIGHT_NEIGHBORHOOD = [(-1, 0), (+1, 0), (0, -1), (0, +1), (-1, -1), (-1, +1), (+1, +1), (+1, -1)]

    if neighborhood == 'eight':
        NEIGHBORHOOD = EIGHT_NEIGHBORHOOD
    else:
        NEIGHBORHOOD = FOUR_NEIGHBORHOOD

    height, width, _ = z.shape
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)
    if debug:
        print 'beta', beta

    vert_shifted_up = z - np.roll(z, 1, axis=0)  # (i,j) gives norm(z[i,j] - z[i-1,j])
    vert_shifted_down = z - np.roll(z, -1, axis=0)  # (i,j) gives norm(z[i,j] - z[i+1,j])

    horiz_shifted_left = z - np.roll(z, 1, axis=1)  # (i,j) gives norm(z[i,j] - z[i,j-1])
    horiz_shifted_right = z - np.roll(z, -1, axis=1)  # (i,j) gives norm(z[i,j] - z[i,j+1])

    energies = []
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_up, vert_shifted_up), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_down, vert_shifted_down), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_left, horiz_shifted_left), axis=2)))
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_right, horiz_shifted_right), axis=2)))

    # Diagnonal components
    if neighborhood == 'eight':
        nw = z - np.roll(np.roll(z, 1, axis=0), 1, axis=1)  # (i,j) gives norm(z[i,j] - z[i-1,j-1])
        ne = z - np.roll(np.roll(z, 1, axis=0), -1, axis=1)  # (i,j) gives norm(z[i,j] - z[i-1,j+1])
        se = z - np.roll(np.roll(z, -1, axis=0), -1, axis=1)  # (i,j) gives norm(z[i,j] - z[i+1,j+1])
        sw = z - np.roll(np.roll(z, -1, axis=0), 1, axis=1)  # (i,j) gives norm(z[i,j] - z[i+1,j-1])

        energies.append(np.exp(-1 * beta * np.sum(np.multiply(nw, nw), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(ne, ne), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(se, se), axis=2)))
        energies.append(np.exp(-1 * beta * np.sum(np.multiply(sw, sw), axis=2)))
    if compute_dict:
        for h in xrange(height):
            for w in xrange(width):
                if (h, w) not in smoothness_matrix:
                    smoothness_matrix[(h, w)] = dict()
                for i, (hh, ww) in enumerate(NEIGHBORHOOD):
                    nh, nw = h + hh, w + ww
                    if nw < 0 or nw >= width:
                        continue
                    if nh < 0 or nh >= height:
                        continue

                    if (nh, nw) not in smoothness_matrix:
                        smoothness_matrix[(nh, nw)] = dict()

                    if (h, w) in smoothness_matrix[(nh, nw)]:
                        continue

                    smoothness_matrix[(h, w)][(nh, nw)] = energies[i][h, w]
                    smoothness_matrix[(nh, nw)][(h, w)] = smoothness_matrix[(h, w)][(nh, nw)]

                    if debug:
                        print (h, w), '->', (nh, nw), ":", z[h, w, :], z[nh, nw, :], smoothness_matrix[(h, w)][(nh, nw)]

        return smoothness_matrix, energies
    return energies


################################################################################
############################## DEBUGGING HELPERS ###############################
################################################################################

# Given an alpha and component map for each pixel, visualizes all the
# components appropriately. All the background clusters are mapped in shades of
# orange, and foregorund clusters are mapped in shades of blue.
#
# img_shape - image dimnesions
# k - matrix of components maps representing which component each pixel belongs
#   to
# alpha - matrix of alpha map, representing if an pixel if fg or bg
# iteration - iteration number for logging purposes
# image_name - image name for logging purposes
def visualize_clusters(img_shape, k, alpha, iteration, image_name, show_image=False, save_image=True):
    BG_COLORS = [[204, 102, 0], [255, 128, 0], [255, 153, 51], [255, 178, 102], [255, 204, 153]]
    FG_COLORS = [[0, 0, 255], [0, 0, 200], [0, 0, 150], [0, 0, 100], [0, 0, 50]]
    res = np.zeros(img_shape, dtype=np.uint8)
    for h in xrange(img_shape[0]):
        for w in xrange(img_shape[1]):
            if alpha[h, w] == 0:
                COLORS = BG_COLORS
            else:
                COLORS = FG_COLORS
            res[h, w, :] = COLORS[k[h, w]]

    target_dir = "output/components/" + image_name

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    plt.imsave(os.path.join(target_dir, "iter_" + str(iteration) + ".png"), res)

    plt.imshow(res)
    plt.show()


# Computes gamma based on entropy of the spectral histogram of the given
# image.
#
# z - image pixels
# img_name - name for logging purposes
def compute_gamma(z, img_name, debug=False, save_fig=False):
    R, G, B = z[:, :, 0], z[:, :, 1], z[:, :, 2]
    img = z.copy()
    matplotlib.colors.rgb_to_hsv(img)

    H = img[:, :, 0].flatten()
    H = H[np.random.randint(H.shape[0], size=100000)]

    probs = np.zeros((256))
    for i in xrange(256):
        probs[i] = np.sum(H == i)

    probs = probs[probs != 0]
    entropy = -np.sum(np.multiply(probs, np.log2(probs)))
    print "%s: %0.2f" % (img_name, entropy / 10000)

    if savefig:
        plt.figure()
        plt.hist(H, 256, range=(0.0, 255.0), color='k', edgecolor='k')
        plt.xlabel("%s: %0.2f" % (img_name, entropy / 10000))
        plt.savefig('hists/' + img_name + '-h.eps', bbox_inches='tight')
        plt.close()


# Grabcut loop
# This function contains the actual implementation of the entire grabcut
# algorithm
#
# img - image to perform segmentation on
# image_name - image name for logging purposes
# user_interaction - boolean specifing if user interaction should be enabled
# num_iterations - number of iterations to run grabcut for
# num_components - number of components to inititalize the fg/bg GMM with
# get_all_segmentations - Stores and returns the intermediate segmentation from
#   each iteration for experimental purposes
def grabcut(img, saliency, image_name, user_interaction=False, num_iterations=10,
            num_components=5, get_all_segmentations=False, debug=False, drawImage=False,
            visualize_clusters=False):
    if debug:
        print 'Initializing gmms'
        tic()
    alpha, foreground_gmm, background_gmm = initialization(img, saliency, num_components=num_components)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    if debug:
        toc('Initializing gmms')

    if debug:
        print 'Computing smoothness matrix...'
        tic()

    pairwise_energies = compute_smoothness_vectorized(img, neighborhood='eight', debug=False)

    if debug:
        toc('Computing smoothness matrix')

    if debug:
        print 'Starting EM'

    segmentations = []
    segmentations.append(alpha)
    user_definite_background = set()
    pixels = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    for user_interaction_iteration in xrange(2):
        for iteration in xrange(1, num_iterations + 1):
            if debug:
                print '----------------------------------------------'
                print 'Iteration %d' % iteration
                print np.sum(alpha) / float(img.shape[0] * img.shape[1])

            # 1. Assigning GMM components to pixels
            if debug:
                tic()
            foreground_components = foreground_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))
            background_components = background_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))

            k = np.ones((img.shape[0], img.shape[1]), dtype=int) * -1
            k[alpha == 1] = foreground_components[alpha == 1]
            k[alpha == 0] = background_components[alpha == 0]

            if debug:
                toc('Assigning GMM components')

            # Cluster visualization
            if visualize_clusters:
                visualize_clusters(img.shape, k, alpha, iteration, image_name,
                                   show_image=True, save_image=False)

            # 2. Learn GMM parameters
            if debug:
                tic()
            foreground_assignments = -1 * np.ones(k.shape)
            foreground_assignments[alpha == 1] = k[alpha == 1]

            background_assignments = -1 * np.ones(k.shape)
            background_assignments[alpha == 0] = k[alpha == 0]

            foreground_gmm.update_components(img, foreground_assignments)
            background_gmm.update_components(img, background_assignments)

            if debug:
                toc('Updating GMM parameters')

            # 3. Estimate segmentation using min cut
            # Compute Unary weights
            if debug:
                tic()
            graph = create_graph(img)
            theta = (background_gmm, foreground_gmm)

            foreground_energies = get_unary_energy_vectorized(1, foreground_components.reshape(
                (img.shape[0] * img.shape[1], 1)), theta, pixels)
            background_energies = get_unary_energy_vectorized(0, background_components.reshape(
                (img.shape[0] * img.shape[1], 1)), theta, pixels)

            done_with = set()
            upper = np.percentile(saliency, 95)
            lower = np.percentile(saliency, 30)
            for h in xrange(img.shape[0]):
                for w in xrange(img.shape[1]):
                    index = h * img.shape[1] + w
                    # If pixel is outside of bounding box, assign large unary energy

                    if h < 2 or w < 2 or img.shape[0] - h < 2 or img.shape[1] - w < 2:
                        w1 = 1e9
                        w2 = 0

                    elif (w, h) in user_definite_background:
                        w1 = 1e9
                        w2 = 0
                    else:
                        # Source: Compute U for curr node
                        start_time = time.time()
                        w1 = foreground_energies[index]  # to background node
                        w2 = background_energies[index]  # to foreground node

                    graph.add_tedge(index, w1, w2)

            # Compute pairwise weights
            NEIGHBORHOOD = [(-1, 0), (+1, 0), (0, -1), (0, +1), (-1, -1), (-1, +1), (+1, +1), (+1, -1)]
            src_h = np.tile(np.arange(img.shape[0]).reshape(img.shape[0], 1), (1, img.shape[1]))
            src_w = np.tile(np.arange(img.shape[1]).reshape(1, img.shape[1]), (img.shape[0], 1))
            src_h = src_h.astype(np.int32)
            src_w = src_w.astype(np.int32)

            for i, energy in enumerate(pairwise_energies):
                if i in [1, 3, 6, 7]:
                    continue
                height_offset, width_offset = NEIGHBORHOOD[i]

                dst_h = src_h + height_offset
                dst_w = src_w + width_offset

                idx = np.logical_and(np.logical_and(dst_h >= 0, dst_h < img.shape[0]),
                                     np.logical_and(dst_w >= 0, dst_w < img.shape[1]))

                src_idx = src_h * img.shape[1] + src_w
                dst_idx = dst_h * img.shape[1] + dst_w

                src_idx = src_idx[idx].flatten()
                dst_idx = dst_idx[idx].flatten()
                weights = energy.astype(np.float32)[idx].flatten()
                weights = gamma * weights

                for l in range(len(src_idx)):
                    graph.add_edge(src_idx[l], dst_idx[l], weights[l], weights[l])
                    # graph.add_edge_vectorized(src_idx, dst_idx, weights, weights)

            if debug:
                toc("Creating graph")

            # Graph has been created, run minCut
            if debug:
                tic()
            graph.maxflow()

            partition = np.array([graph.get_segment(i) for i in range(graph.get_node_num())])

            if debug:
                toc("Min cut")

            # Update alpha
            if debug:
                tic()

            partition = partition.reshape(alpha.shape)
            num_changed_pixels = np.sum(np.abs(partition - alpha))
            alpha = partition
            segmentations.append(alpha)

            if debug:
                toc("Updating alphas")

            relative_change = num_changed_pixels / float(img.shape[0] * img.shape[1])

            if drawImage:
                if iteration % 10 == 0 or (iteration == num_iterations and not user_interaction):
                    result = np.reshape(partition, (img.shape[0], img.shape[1])) * 255
                    result = result.astype(dtype=np.uint8)
                    result = np.dstack((result, result, result))
                    plt.imshow(result)
                    plt.show()
            if debug:
                print 'Relative change was %f' % relative_change

        # Prompt for user interaction if enabled
        if user_interaction:
            user_img = img.copy()
            user_img[alpha == 0] = 0
            points = get_user_polyline(user_img)
            # Add to "definite background" set
            neighborhood_offsets = range(-5, 6)
            for (x, y) in points:
                for xxx in neighborhood_offsets:
                    for yyy in neighborhood_offsets:
                        current_x = int(x + xxx)
                        current_y = int(y + yyy)
                        if current_x < 0 or current_y < 0 or current_x > img.shape[0] or current_y > img.shape[1]:
                            continue
                        user_definite_background.add((current_x, current_y))
                        alpha[current_y, current_x] = 0
        else:
            break

    if get_all_segmentations:
        return segmentations
    else:
        return alpha


def load_saliency(filename):
    map = np.load(filename)
    return map


def main():
    args = get_args()
    img = load_image(args.image_file)

    saliency = load_saliency(args.saliency_file)
    print '----------------------------------------------'
    print 'Running GrabCut with the Following parameters:'
    print 'Image: %s' % args.image_file
    print 'Saliency: %s' % args.saliency_file
    print 'Num Iterations: %d' % args.num_iterations
    print 'Num Components: %d' % args.num_components
    print 'User Interaction Enabled: %r' % args.user_interaction
    print '----------------------------------------------'

    grabcut(img, saliency, args.image_file, num_iterations=args.num_iterations,
            num_components=args.num_components, user_interaction=args.user_interaction,
            debug=True, drawImage=True)


################################################################################
######################## UNVECTORIZED GRABCUT HELPERS ##########################
############################### NOT USED IN CODE ###############################
################################################################################

# Helper function for computing pairwise energies
# Returns the weight of each component in the given GMM
def get_pi(alpha, k, gmms):
    return gmms[alpha].weights[k]


# Helper function for computing pairwise energies
# Returns the determinant of the covariance matrix of the given GMM
def get_cov_det(alpha, k, gmms):
    return gmms[alpha].gaussians[k].sigma_det


# Helper function for computing pairwise energies
# Returns the mean of the given GMM
def get_mean(alpha, k, gmms):
    return gmms[alpha].gaussians[k].mean


# Helper function for computing pairwise energies
# Returns the inverse of the covariance matrix of the given GMM
def get_cov_inv(alpha, k, gmms):
    return gmms[alpha].gaussians[k].sigma_inv


# Helper function for computing pairwise energies
# Its not log_prob but we are calling it that for convinience
# Computes part of the probabilty of the pixel belonging to a specific component
# of the given GMM
def get_log_prob(alpha, k, gmms, z_pixel):
    term = (z_pixel - get_mean(alpha, k, gmms))
    return 0.5 * np.dot(np.dot(term.T, get_cov_inv(alpha, k, gmms)), term)


# Given an image (z), computes the expected difference between neighboring
# pixels, and returns the corresponding beta value.
def compute_beta(z, debug=False):
    accumulator = 0
    m = z.shape[0]
    n = z.shape[1]

    for h in xrange(m - 1):
        if debug: print 'Computing row', h
        for w in xrange(n):
            accumulator += np.linalg.norm(z[h, w, :] - z[h + 1, w, :]) ** 2

    for h in xrange(m):
        if debug: print 'Computing row', h
        for w in xrange(n - 1):
            accumulator += np.linalg.norm(z[h, w, :] - z[h, w + 1, :]) ** 2

    num_comparisons = float(2 * (m * n) - m - n)

    beta = (2 * (accumulator / num_comparisons)) ** -1

    return beta


# Given an image, and an optional neighborhood parameter, computes all the
# pairwise weights between neigboring pixels
#
# z - matrix of image pixels
# neighborhood - 'eight' for 8 neighborhood, 'four' for 4 neighborhood
def compute_smoothness(z, neighborhood='eight', debug=False):
    EIGHT_NEIGHBORHOOD = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    FOUR_NEIGHBORHOOD = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    if neighborhood == 'eight':
        NEIGHBORHOOD = EIGHT_NEIGHBORHOOD
    else:
        NEIGHBORHOOD = FOUR_NEIGHBORHOOD

    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)
    if debug:
        print 'beta', beta

    for h in xrange(height):
        if debug:
            print 'Computing row', h
        for w in xrange(width):
            if (h, w) not in smoothness_matrix:
                smoothness_matrix[(h, w)] = dict()
            for hh, ww in NEIGHBORHOOD:
                nh, nw = h + hh, w + ww
                if nw < 0 or nw >= width:
                    continue
                if nh < 0 or nh >= height:
                    continue

                if (nh, nw) not in smoothness_matrix:
                    smoothness_matrix[(nh, nw)] = dict()

                if (h, w) in smoothness_matrix[(nh, nw)]:
                    continue

                smoothness_matrix[(h, w)][(nh, nw)] = \
                    np.exp(-1 * beta * (np.linalg.norm(z[h, w, :] - z[nh, nw, :]) ** 2))
                smoothness_matrix[(nh, nw)][(h, w)] = smoothness_matrix[(h, w)][(nh, nw)]

                if debug:
                    print (h, w), '->', (nh, nw), ":", z[h, w, :], z[nh, nw, :], smoothness_matrix[(h, w)][(nh, nw)]

    return smoothness_matrix


# Given an alpha map, a components map, the GMMS and the image, computes the
# total energy on the image
#
# Currently not used in code
def get_energy(alpha, k, gmms, z, smoothness_matrix):
    # Compute U
    U = 0
    for h in xrange(z.shape[0]):
        for w in xrange(z.shape[1]):
            U += -np.log(get_pi(alpha[h, w], k[h, w], gmms)) \
                 + 0.5 * np.log(get_cov_det(alpha[h, w], k[h, w], gmms)) \
                 + get_log_prob(alpha[h, w], k[h, w], gmms, z[h, w, :])

    # Compute V
    V = 0
    for h in xrange(z.shape[0]):
        for w in xrange(z.shape[1]):
            # Loop through neighbors
            for (nh, nw) in smoothness_matrix[(h, w)].keys():
                if alpha[h, w] != alpha[nh, nw]:
                    V += smoothness_matrix[(h, w)][(nh, nw)]
    V = gamma * V

    return U + V


# Computes the unary energy for a given pixel and GMM
#
# Currently not used in code
def get_unary_energy(alpha, k, gmms, z, pixel):
    h, w = pixel
    return -np.log(get_pi(alpha, k[h, w], gmms)) \
           + 0.5 * np.log(get_cov_det(alpha, k[h, w], gmms)) \
           + get_log_prob(alpha, k[h, w], gmms, z[h, w, :])


# Computes the pairwise energy for two given pixels
#
# Currently not used in code
def get_pairwise_energy(alpha, pixel_1, pixel_2, smoothness_matrix):
    (h, w) = pixel_1
    (nh, nw) = pixel_2
    V = smoothness_matrix[(h, w)][(nh, nw)]

    return gamma * V


# Computes the pairwise energies using the vectorized code, the unvectorized
# code and compares them to make sure both return the same result
def verify_smoothness_matrix(img):
    smoothness_matrix1, _ = compute_smoothness_vectorized(img, neighborhood='eight', compute_dict=True, debug=False)
    smoothness_matrix2 = compute_smoothness(img, neighborhood='eight', debug=False)

    if len(s1) != len(smoothness_matrix):
        print 'PROBLEM lens not equal'
        return False
    for (h, w) in s1:
        if len(s1[(h, w)]) != len(smoothness_matrix[(h, w)]):
            print 'PROBLEM lens not equal at', (h, w)
            return False
        for (nh, nw) in s1[(h, w)]:
            if abs(s1[(h, w)][(nh, nw)] - smoothness_matrix[(h, w)][(nh, nw)]) > 1e-12:
                print 'PROBLEM at', (h, w), (nh, nw), '->', s1[(h, w)][(nh, nw)], '!=', smoothness_matrix[(h, w)][
                    (nh, nw)]
                return False
    print 'Finished check'
    return True


if __name__ == '__main__':
    main()
