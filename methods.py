import utils
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils.extmath import density
import logging
from time import time
from sklearn import metrics
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
