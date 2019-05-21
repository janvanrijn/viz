import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import typing


# https://becominghuman.ai/paper-repro-learning-to-learn-by-gradient-descent-by-gradient-descent-6e504cc1c0de
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='boston')
    parser.add_argument('--x_feature', type=str, default='RM')
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def linear_regression_predict(x: np.array, theta_0: float, theta_b: float) -> np.array:
    return theta_0 * x + theta_b


def loss_score(x: np.array, y_true: np.array, theta_0: float, theta_b: float) -> float:
    y_hat = linear_regression_predict(x, theta_0, theta_b)
    if y_hat.shape != y_true.shape:
        raise ValueError()
    if len(y_hat.shape) > 1:
        raise ValueError()
    return sum((y_true - y_hat) ** 2) / len(y_true)


def delta_theta_0(x: np.array, y_true: np.array, theta_0: float, theta_b: float) -> float:
    y_hat = linear_regression_predict(x, theta_0, theta_b)
    if y_hat.shape != y_true.shape:
        raise ValueError()
    if len(y_hat.shape) > 1:
        raise ValueError()
    errors = -2 * x * (y_true - y_hat)
    return sum(errors) / len(y_true)


def delta_theta_b(x: np.array, y_true: np.array, theta_0: float, theta_b: float) -> float:
    y_hat = linear_regression_predict(x, theta_0, theta_b)
    if y_hat.shape != y_true.shape:
        raise ValueError()
    if len(y_hat.shape) > 1:
        raise ValueError()
    errors = -2 * (y_true - y_hat)
    return sum(errors) / len(y_true)


def linear_regression_sgd(x: np.array, y: np.array, alpha: float, delta: float) -> typing.Tuple[typing.List, typing.List, typing.List]:
    # http://mccormickml.com/2014/03/04/gradient-descent-derivation/
    theta_0 = [1.0]
    theta_b = [0.0]
    loss = [loss_score(x, y, theta_0[-1], theta_b[-1])]
    logging.info('Initial values, theta_0 = %f, theta_b = %f, loss = %f' % (theta_0[-1], theta_b[-1], loss[-1]))

    while len(loss) <= 10 or loss[-1] < loss[-2] - delta:
        delta_theta_0_i = delta_theta_0(x, y, theta_0[-1], theta_b[-1])
        delta_theta_b_i = delta_theta_b(x, y, theta_0[-1], theta_b[-1])
        theta_0_i = theta_0[-1] - alpha * delta_theta_0_i
        theta_b_i = theta_b[-1] - alpha * delta_theta_b_i
        loss_i = loss_score(x, y, theta_0_i, theta_b_i)
        logging.info('Step %d; theta_0 = %f (delta = %f), theta_b = %f (delta = %f), loss = %f' % (len(theta_0),
                                                                                                   theta_0[-1],
                                                                                                   delta_theta_0_i,
                                                                                                   theta_b[-1],
                                                                                                   delta_theta_b_i,
                                                                                                   loss[-1]))
        theta_0.append(theta_0_i)
        theta_b.append(theta_b_i)
        loss.append(loss_i)
    return theta_0, theta_b, loss


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

    # perform SGD
    theta_0, theta_b, loss = linear_regression_sgd(X, y, args.alpha, args.delta)

    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    ax1.scatter(X, y)
    ax1.set_xlabel(args.x_feature)
    ax1.set_ylabel(bunch.details['default_target_attribute'])
    ax1.set_title(args.dataset_name)

    line_x = np.array([min(X), max(X)])
    line_y = linear_regression_predict(line_x, theta_0[-1], theta_b[-1])
    ax1.plot(line_x, line_y, color='red')

    plt.show()


if __name__ == '__main__':
    run(parse_args())
