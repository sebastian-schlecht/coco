import theano.tensor as T

import numpy as np


def berhu(predictions, targets, s=0.2, bounded=False, lower_bound=0.5, upper_bound=1.2):
    """
    Reverse huber loss for high-dimensional regression
    :param predictions:
    :param targets:
    :param s:
    :param bounded:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    if bounded:
        mask = T.gt(targets, lower_bound) * T.lt(targets, upper_bound)
        # Compute n of valid pixels
        n_valid = T.sum(mask)
        # Redundant mult here
        r = (predictions - targets) * mask
    else:
        n_valid = T.prod(targets.shape)
        r = (predictions - targets)
    c = s * T.max(T.abs_(r))
    a_r = T.abs_(r)
    b = T.switch(T.lt(a_r, c), a_r, ((r ** 2) + (c ** 2)) / (2 * c))
    return T.sum(b) / n_valid


def mse(predictions, targets, bounded=False, lower_bound=0., upper_bound=1.2):
    """
    Bounded MSE
    :param predictions:
    :param targets:
    :param bounded:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    if bounded:
        mask = T.gt(targets, lower_bound) * T.lt(targets, upper_bound)
        # Compute n of valid pixels
        n_valid = T.sum(mask)
        # Apply mask
        d = (predictions - targets) * mask
    else:
        n_valid = T.prod(targets.shape)
        d = (predictions - targets)
    return T.sum((d) ** 2) / n_valid


def spatial_gradient(prediction, target):
    """
    Spatial gradient loss according to http://cs.nyu.edu/~deigen/dnl/
    :param prediction:
    :param target:
    :return:
    """
    pred = prediction
    pred_v = pred.flatten(2)
    target_v = target.flatten(2)
    # Compute mask
    mask = T.gt(target_v, 0.)
    # Compute n of valid pixels
    n_valid = T.sum(mask, axis=1)
    # Apply mask and log transform
    m_pred = pred_v * mask
    m_t = T.switch(mask, T.log(target_v), 0.)
    d = m_pred - m_t

    # Define scale invariant cost
    scale_invariant_cost = (T.sum(n_valid * T.sum(d ** 2, axis=1)) * T.sum(T.sum(d, axis=1) ** 2)) / T.maximum(
        T.sum(n_valid ** 2), 1)

    # Add spatial gradient components from D. Eigen DNL

    # Squeeze in case
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
    if target.ndim == 4:
        target = target[:, 0, :, :]
    # Mask in tensor form
    mask_tensor = T.gt(target, 0.)
    # Project into log space
    target = T.switch(mask_tensor, T.log(target), 0.)
    # Stepsize
    h = 1
    # Compute spatial gradients symbolically
    p_di = (pred[:, h:, :] - pred[:, :-h, :]) * (1 / np.float32(h))
    p_dj = (pred[:, :, h:] - pred[:, :, :-h]) * (1 / np.float32(h))
    t_di = (target[:, h:, :] - target[:, :-h, :]) * (1 / np.float32(h))
    t_dj = (target[:, :, h:] - target[:, :, :-h]) * (1 / np.float32(h))
    m_di = T.and_(mask_tensor[:, h:, :], mask_tensor[:, :-h, :])
    m_dj = T.and_(mask_tensor[:, :, h:], mask_tensor[:, :, :-h])
    # Define spatial grad cost
    grad_cost = T.sum(m_di * (p_di - t_di) ** 2) / T.sum(m_di) + \
        T.sum(m_dj * (p_dj - t_dj) ** 2) / T.sum(m_dj)
    # Compute final expression
import theano.tensor as T

import numpy as np


def berhu(predictions, targets, s=0.2, bounded=False, lower_bound=0.5, upper_bound=1.2):
    """
    Reverse huber loss for high-dimensional regression
    :param predictions:
    :param targets:
    :param s:
    :param bounded:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    if bounded:
        mask = T.gt(targets, lower_bound) * T.lt(targets, upper_bound)
        # Compute n of valid pixels
        n_valid = T.sum(mask)
        # Redundant mult here
        r = (predictions - targets) * mask
    else:
        n_valid = T.prod(targets.shape)
        r = (predictions - targets)
    c = s * T.max(T.abs_(r))
    a_r = T.abs_(r)
    b = T.switch(T.lt(a_r, c), a_r, ((r ** 2) + (c ** 2)) / (2 * c))
    return T.sum(b) / n_valid


def mse(predictions, targets, bounded=False, lower_bound=0., upper_bound=1.2):
    """
    Bounded MSE
    :param predictions:
    :param targets:
    :param bounded:
    :param lower_bound:
    :param upper_bound:
    :return:
    """
    if bounded:
        mask = T.gt(targets, lower_bound) * T.lt(targets, upper_bound)
        # Compute n of valid pixels
        n_valid = T.sum(mask)
        # Apply mask
        d = (predictions - targets) * mask
    else:
        n_valid = T.prod(targets.shape)
        d = (predictions - targets)
    return T.sum((d) ** 2) / n_valid


def spatial_gradient(prediction, target):
    """
    Spatial gradient loss according to Eigen DNL
    :param prediction:
    :param target:
    :return:
    """
    pred = prediction
    pred_v = pred.flatten(2)
    target_v = target.flatten(2)
    # Compute mask
    mask = T.gt(target_v, 0.)
    # Compute n of valid pixels
    n_valid = T.sum(mask, axis=1)
    # Apply mask and log transform
    m_pred = pred_v * mask
    m_t = T.switch(mask, T.log(target_v), 0.)
    d = m_pred - m_t

    # Define scale invariant cost
    scale_invariant_cost = (T.sum(n_valid * T.sum(d ** 2, axis=1)) * T.sum(T.sum(d, axis=1) ** 2)) / T.maximum(
        T.sum(n_valid ** 2), 1)

    # Add spatial gradient components from D. Eigen DNL

    # Squeeze in case
    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
    if target.ndim == 4:
        target = target[:, 0, :, :]
    # Mask in tensor form
    mask_tensor = T.gt(target, 0.)
    # Project into log space
    target = T.switch(mask_tensor, T.log(target), 0.)
    # Stepsize
    h = 1
    # Compute spatial gradients symbolically
    p_di = (pred[:, h:, :] - pred[:, :-h, :]) * (1 / np.float32(h))
    p_dj = (pred[:, :, h:] - pred[:, :, :-h]) * (1 / np.float32(h))
    t_di = (target[:, h:, :] - target[:, :-h, :]) * (1 / np.float32(h))
    t_dj = (target[:, :, h:] - target[:, :, :-h]) * (1 / np.float32(h))
    m_di = T.and_(mask_tensor[:, h:, :], mask_tensor[:, :-h, :])
    m_dj = T.and_(mask_tensor[:, :, h:], mask_tensor[:, :, :-h])
    # Define spatial grad cost
    grad_cost = T.sum(m_di * (p_di - t_di) ** 2) / T.sum(m_di) + \
        T.sum(m_dj * (p_dj - t_dj) ** 2) / T.sum(m_dj)
    # Compute final expression
    return scale_invariant_cost + grad_cost
    return scale_invariant_cost + grad_cost
