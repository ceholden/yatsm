""" Plots useful for YATSM
"""
from datetime import datetime as dt
import logging
import re

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger('yatsm')


def plot_crossvalidation_scores(kfold_scores, test_labels):
    """ Plots KFold test summary statistics

    Args:
      kfold_scores (np.ndarray): n by 2 shaped array of mean and standard
        deviation of KFold scores
      test_labels (list): n length list of KFold label names

    """
    return
    ind = np.arange(kfold_scores.shape[0])
    width = 0.5

    fig, ax = plt.subplots()
    bars = ax.bar(ind, kfold_scores[:, 0], width)
    _, caplines, _ = ax.errorbar(ind + width / 2.0, kfold_scores[:, 0],
                                 fmt='none',
                                 yerr=kfold_scores[:, 1],
                                 capsize=10, elinewidth=3)
    for capline in caplines:
        capline.set_linewidth(10)
        capline.set_markeredgewidth(3)
        capline.set_color('red')

    for i, bar in enumerate(bars):
        txt = r'%.3f $\pm$ %.3f' % (kfold_scores[i, 0], kfold_scores[i, 1])
        ax.text(ind[i] + width / 2.0,
                kfold_scores[i, 0] / 2.0,
                txt,
                ha='center', va='bottom', size='large')

    ax.set_xticks(ind + width / 2.0)
    ax.set_xticklabels(test_labels, ha='center')
    # plt.ylim((0, 1.0))

    plt.title('KFold Cross Validation Summary Statistics')
    plt.xlabel('Test')
    plt.ylabel(r'Accuracy ($\pm$ standard deviation)')

    plt.tight_layout()
    plt.show()


def plot_feature_importance(algo, cfg):
    """ Plots Random Forest feature importance as barplot

    If YATSM configuration (cfg['YATSM']) contains Patsy design information in
    'design_info', then this plot will show more specific labels for each
    feature.

    Args:
      algo (sklearn.ensemble.RandomForestClassifier): Random Forest algorithm
      cfg (dict): YATSM configuration dictionary

    """
    ind = np.arange(algo.feature_importances_.size)
    width = 0.5

    n_bands = cfg['dataset']['n_bands'] - 1

    # Form betas from design matrix
    if 'design' not in cfg['YATSM']:
        logger.warning('Design info not provided to plot -- will use basic '
                       'coefficient labels')
        # First remove out RMSE  # TODO: check if we used RMSE
        n_feat = algo.feature_importances_.size - n_bands
        # All bands have same n_coef, so leftover are other variables
        betas = range(0, int(n_feat / n_bands))
    else:
        # Rename slope, convert harm.*[0] to cos, and harm.*[1] to sin
        betas = []
        for _beta in cfg['YATSM']['design'].keys():
            _beta = re.sub(r'^x$', 'slope', _beta)
            _beta = re.sub(r'harm(.*)\[0\]', r'cos\1', _beta)
            _beta = re.sub(r'harm(.*)\[1\]', r'sin\1', _beta)

            betas.append(_beta)
    betas.append('RMSE')  # TODO: check if we used RMSE

    # Grab bands
    bands = range(1, cfg['dataset']['n_bands'] + 1)
    bands.remove(cfg['dataset']['mask_band'])  # band is now index so + 1

    # names = [r'Band %i $\beta_{%s}$' % (b, i)
    #          for i in betas for b in bands]
    # names += [r'Band %i $RMSE$' % b for b in bands]
    names = ['Band %i' % b for b in bands * (len(betas) + 1)]

    fig, ax = plt.subplots()
    ax.bar(ind, algo.feature_importances_, width)
    ax.set_xticks(ind + width / 2.0)
    ax.set_xticklabels(names, ha='center', rotation=90)
    for i_b in ind[::n_bands]:
        ax.axvline(i_b, c='k', lw=2)
    for i, _beta in enumerate(betas):
        _x = i * len(bands) + len(bands) / 2.0
        ax.annotate(_beta,
                    (_x, 0), xycoords='data',
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', size='large')
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.show()


def plot(yatsm, band, freq, ylim=None):
    """ Plot YATSM results for a specified band
    Args:
      yatsm (yatsm.YATSM): model
      band (int): data band to plot
      freq (iterable): frequency of sine/cosine (for predictions)
      ylim (tuple): tuple of floats for y-axes limits

    """
    raise Exception('This function is depricated in v0.5.0. Use YATSM object'
                    ' for plotting')
    from utils import make_X

    dates = map(dt.fromordinal, yatsm.X[:, 1].astype(np.uint32))

    # Plot data
    plt.plot(dates, yatsm.Y[band, :], 'ko')

    if ylim:
        plt.ylim(ylim)

    # Add in lines and break points
    for rec in yatsm.record:
        # Create sequence of X between start and end dates
        if rec['start'] < rec['end']:
            mx = np.arange(rec['start'], rec['end'])
        elif rec['start'] > rec['end']:
            mx = np.arange(rec['end'], rec['start'])
        else:
            continue
        mdates = map(dt.fromordinal, mx)

        # Predict
        mX = make_X(mx, freq)
        my = np.dot(rec['coef'][:, 4], mX)

        # Plot prediction
        plt.plot(mdates, my, linewidth=2)

        # Plot change
        if rec['break'] > 0:
            i = np.where(yatsm.X[:, 1] == rec['break'])[0]
            plt.plot(dt.fromordinal(rec['break']),
                     yatsm.Y[band, i],
                     'ro', mec='r', mfc='none', ms=10, mew=5)

    plt.show()
