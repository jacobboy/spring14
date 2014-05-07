import utils
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils.extmath import density
import logging
from time import time
from sklearn import metrics
from matplotlib import pyplot as plt
# <codecell>

def get_data(dataset_version):
    feature_names_comps = utils.load_array("%s/feature_names_comps" % dataset_version)
    feature_names_ab = utils.load_array("%s/feature_names_ab" % dataset_version)
    feature_names_ti = utils.load_array("%s/feature_names_ti" % dataset_version)
    X_comps = utils.load_array("%s/X_comps" % dataset_version)
    X_ab = utils.load_csr("%s/X_ab" % dataset_version)
    X_ti = utils.load_csr("%s/X_ti" % dataset_version)
    y = utils.load_array("%s/y" % dataset_version)
    return feature_names_comps, feature_names_ab, feature_names_ti, X_comps, X_ab, X_ti, y

def matrix_scatter_pcs(names = None, tks = True, **kwargs):
    import itertools
    from pandas import DataFrame as DF
    import pandas as pd
    to_plot = []
    for key, value in kwargs.iteritems():
        if key != "tks":
            if value.shape[0] > value.shape[1]:
                to_plot.append((key, DF(value.T)))
            else:
                to_plot.append((key, DF(value)))
    numvar = to_plot[0][1].shape[0]
    s = range(numvar)
    if names == None:
        names = ["PC " + str(x + 1) for x in s]
    fgr, axs = plt.subplots(numvar, numvar)
    # colors = cm.rainbow(np.linspace(0, 1, numvar))
    colors = ["blue", "red", "green"]
    for xidx, yidx in itertools.product(s, repeat=2):
        ax = axs[yidx,xidx]
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # Set up ticks only on one side for the "edge" subplots...
        if xidx == 0:
            ax.set_ylabel(names[yidx])
            ax.yaxis.set_ticks([])
            ax.yaxis.set_visible(True)
        if xidx == max(s):
            if tks:
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_visible(True)
        if yidx == 0:
            ax.set_xlabel(names[xidx])
            ax.xaxis.set_label_position("top")
            ax.xaxis.set_visible(True)
            ax.xaxis.set_ticks([])
        if yidx == max(s):
            if tks:
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_visible(True)
        xs = []
        ys = []
        i = 0
        for name, plot in to_plot:
            x, y = plot.iloc[xidx,:], plot.iloc[yidx,:]
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y, s = 3, color = colors[i], label = name)
            i+=1
        allxs = pd.concat(xs)
        allys = pd.concat(ys)
        ax.set_xlim([min(allxs), max(allxs)])
        ax.set_ylim([min(allys), max(allys)])
    return fgr, axs
