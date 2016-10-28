"""
Credits: source code borrowed & adapted from https://github.com/ahaque/cs231b
"""
import numpy as np
from gmm import GMM
import maxflow

# Global constants
gamma = 50

# If energy changes less than CONVERGENCE_CRITERIA percent from the last iteration
# we will terminate
CONVERGENCE_CRITERION = 0.02


# Given an image and bounding box, initializes a foreground and a background
# GMM. The number of components can optionally be passed in.
def initialization(img, saliency, num_components=5):
    height, width, _ = img.shape
    alpha = np.zeros((height, width), dtype=np.int8)

    upper = np.percentile(saliency, 95)
    lower = np.percentile(saliency, 30)

    assert img.shape[:2] == saliency.shape[:2]

    alpha[2:-2, 2:-2] = 1

    foreground_gmm = GMM(num_components)
    background_gmm = GMM(num_components)

    foreground_gmm.initialize_gmm(img[saliency > upper])
    background_gmm.initialize_gmm(img[saliency < lower])

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
def get_total_unary_energy_vectorized(gmm, pixels):
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
def get_unary_energy_vectorized(alpha, k, gmms, pixels):
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
    return -np.log(pi) \
           + 0.5 * np.log(dets) \
           + 0.5 * log_prob


# Given an image (z), computes the expected difference between neighboring
# pixels, and returns the corresponding beta value.
def compute_beta_vectorized(z):
    m = z.shape[0]
    n = z.shape[1]

    vert_shifted = z - np.roll(z, 1, axis=0)
    temp = np.sum(np.multiply(vert_shifted, vert_shifted), axis=2)
    accumulator = np.sum(temp[1:, :])

    horiz_shifted = z - np.roll(z, 1, axis=1)
    temp = np.sum(np.multiply(horiz_shifted, horiz_shifted), axis=2)
    accumulator += np.sum(temp[:, 1:])

    num_comparisons = float(2 * (m * n) - m - n)
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
def compute_smoothness_vectorized(z, neighborhood='eight', compute_dict=False):
    FOUR_NEIGHBORHOOD = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    EIGHT_NEIGHBORHOOD = [(-1, 0), (+1, 0), (0, -1), (0, +1), (-1, -1), (-1, +1), (+1, +1), (+1, -1)]

    if neighborhood == 'eight':
        NEIGHBORHOOD = EIGHT_NEIGHBORHOOD
    else:
        NEIGHBORHOOD = FOUR_NEIGHBORHOOD

    height, width, _ = z.shape
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)

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

        return smoothness_matrix, energies
    return energies


# Grabcut loop
# This function contains the actual implementation of the entire grabcut
# algorithm using saliency seed points to estimate the GMM colour model
def saliency_grabcut(img, saliency, num_iterations=10,
                     num_components=5, get_all_segmentations=False):
    alpha, foreground_gmm, background_gmm = initialization(img, saliency, num_components=num_components)
    k = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    pairwise_energies = compute_smoothness_vectorized(img, neighborhood='eight')

    segmentations = []
    segmentations.append(alpha)
    user_definite_background = set()
    pixels = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    for user_interaction_iteration in xrange(2):
        for iteration in xrange(1, num_iterations + 1):
            foreground_components = foreground_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))
            background_components = background_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))

            k = np.ones((img.shape[0], img.shape[1]), dtype=int) * -1
            k[alpha == 1] = foreground_components[alpha == 1]
            k[alpha == 0] = background_components[alpha == 0]

            foreground_assignments = -1 * np.ones(k.shape)
            foreground_assignments[alpha == 1] = k[alpha == 1]

            background_assignments = -1 * np.ones(k.shape)
            background_assignments[alpha == 0] = k[alpha == 0]

            foreground_gmm.update_components(img, foreground_assignments)
            background_gmm.update_components(img, background_assignments)

            graph = create_graph(img)
            theta = (background_gmm, foreground_gmm)

            foreground_energies = get_unary_energy_vectorized(1, foreground_components.reshape(
                (img.shape[0] * img.shape[1], 1)), theta, pixels)
            background_energies = get_unary_energy_vectorized(0, background_components.reshape(
                (img.shape[0] * img.shape[1], 1)), theta, pixels)

            done_with = set()
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

            graph.maxflow()

            partition = np.array([graph.get_segment(i) for i in range(graph.get_node_num())])

            partition = partition.reshape(alpha.shape)
            alpha = partition
            segmentations.append(alpha)
        else:
            break

    if get_all_segmentations:
        return segmentations
    else:
        return alpha
