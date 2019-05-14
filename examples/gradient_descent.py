import argparse
import logging
import matplotlib.pyplot as plt
import sklearn.datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='boston')
    parser.add_argument('--x_feature', type=str, default='RM')
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


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
    plt.show()


if __name__ == '__main__':
    run(parse_args())
