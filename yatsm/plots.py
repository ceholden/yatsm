""" Plots useful for YATSM
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_crossvalidation_scores(kfold_scores, test_labels):
    """ Plots KFold test summary statistics

    Args:
      kfold_scores (np.ndarray): n by 2 shaped array of mean and standard
        deviation of KFold scores
      test_labels (list): n length list of KFold label names

    """
    ind = np.arange(kfold_scores.shape[0])
    width = 0.5

    fig, ax = plt.subplots()
    ax.bar(ind, kfold_scores[:, 0], width, yerr=kfold_scores[:, 1])
    ax.set_xticks(ind + width / 2.0)
    ax.set_xticklabels(test_labels, ha='center')
    plt.ylim((0, 1.0))

    plt.title('KFold Cross Validation Summary Statistics')
    plt.xlabel('Test')
    plt.ylabel(r'Accuracy ($\pm$ standard deviation)')

    plt.show()


def plot_feature_importance(algo, dataset_config, yatsm_config):
    """ Plots Random Forest feature importance as barplot

    Args:
      algo (sklearn.ensemble.RandomForestClassifier): Random Forest algorithm
      dataset_config (dict): dataset configuration details
      yatsm_config (dict): YATSM model run details

    """
    ind = np.arange(algo.feature_importances_.size)
    width = 0.5

    betas = range(0, 2 + 2 * len(yatsm_config['freq']))
    bands = range(1, dataset_config['n_bands'] + 1)
    bands.remove(dataset_config['mask_band'] + 1)  # band is now index so + 1

    names = [r'Band {b} $\beta_{i}$'.format(b=b, i=i)
             for i in betas for b in bands]
    names += [r'Band {b} $RMSE$'.format(b=b) for b in bands]

    fig, ax = plt.subplots()
    ax.bar(ind, algo.feature_importances_, width)
    ax.set_xticks(ind + width / 2.0)
    ax.set_xticklabels(names, rotation=90, ha='center')
    ax.vlines(ind[::dataset_config['n_bands'] - 1],
              algo.feature_importances_.min(),
              algo.feature_importances_.max())
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.show()
