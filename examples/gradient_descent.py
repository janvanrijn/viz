import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='boston')
    parser.add_argument('--x_feature', type=str, default='RM')
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def linear_regression_sgd(x: np.array, y: np.array):
    # http://mccormickml.com/2014/03/04/gradient-descent-derivation/
    theta_0 = [0.0]
    theta_b = [1.0]
    pass


def linear_regression_predict(x: np.array, theta_0: float, theta_b: float):
    return theta_0 * x + theta_b


def loss(y_hat: np.array, y_true: np.array):
    if y_hat.shape != y_true.shape:
        raise ValueError()
    if len(y_hat.shape) > 1:
        raise ValueError()
    sum_errors = 0.0
    for y_hat_i, y_true_i in zip(y_hat, y_true):
        sum_errors += (y_hat_i - y_true_i) ** 2
    return sum_errors / len(y_hat)


def delta_loss(y_hat, y_true):
    if y_hat.shape != y_true.shape:
        raise ValueError()
    if len(y_hat.shape) > 1:
        raise ValueError()
    sum_errors = 0.0
    for y_hat_i, y_true_i in zip(y_hat, y_true):
        sum_errors += y_hat_i - y_true_i
    return 2 / len(y_hat) * sum_errors


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    bunch = sklearn.datasets.fetch_openml(args.dataset_name)
    feature_name_idx = {feat: idx for idx, feat in enumerate(bunch.feature_names)}
    X = bunch.data[:, feature_name_idx[args.x_feature]]
    logging.info('Dataset: %s, features: %s' % (bunch.details['name'], feature_name_idx))
    y = bunch.target

    if X.shape != y.shape:
        raise ValueError('Problem with data shapes. X %s, y %s' % (X.shape, y.shape))

    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    ax1.scatter(X, y)
    ax1.set_xlabel(args.x_feature)
    ax1.set_ylabel(bunch.details['default_target_attribute'])
    ax1.set_title(args.dataset_name)
    plt.show()


if __name__ == '__main__':
    run(parse_args())
