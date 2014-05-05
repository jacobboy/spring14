# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
from pandas import Series
from pandas import DataFrame as DF
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
import utils
from utils import trm
from methods import get_data
from classes import Benchmarker
from classes import L1LinearSVC

import logging
from time import time
import numpy as np
import sklearn
import scipy
import pylab as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# <codecell>

# constants
dataset_version = "data2"
select_chi2s=10000
print_topX=50
print_report=True
print_cm=False
save_fig=True

categories = ["1"]

bm = Benchmarker(print_topX, print_report, print_cm)

def benchmark(clf):
    return bm.benchmark(clf)

# <codecell>

################################################################################
# load data

feature_names_comps, feature_names_tfidf, X_train_comps, X_train_tfidf, X_test_comps, X_test_tfidf, y_train, y_test = get_data(dataset_version)


# <codecell>

################################################################################
# extract features via chi2
# assemble X_train, X_test, and feature_names

# if select_chi2:
print("Extracting %d best features by a chi-squared test" %
      select_chi2)
t0 = time()
ch2 = SelectKBest(chi2, k=select_chi2)
X_train_ch2 = ch2.fit_transform(X_train_tfidf, y_train)
X_train = hstack([X_train_comps, X_train_ch2])
X_test_ch2 = ch2.transform(X_test_tfidf)
X_test = hstack([X_test_comps, X_test_ch2])
feature_names_ch2 = feature_names_tfidf[ch2.get_support()]
feature_names = np.append(feature_names_comps, feature_names_ch2)
print("done in %fs" % (time() - t0))
print()

# <codecell>

# Finally, get to work
results = []

# <codecell>

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

# <codecell>

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# <codecell>

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# <codecell>

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# <codecell>

# Input X must be non-negative
# # Train sparse Naive Bayes classifiers
# print('=' * 80)
# print("Naive Bayes")
# results.append(benchmark(MultinomialNB(alpha=.01)))
# results.append(benchmark(BernoulliNB(alpha=.01)))

# <codecell>

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))

# <codecell>

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(6)]

clf_names, score, training_time, test_time, clfs, preds = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

pl.figure(figsize=(12,8))
pl.title("Score for %d best features" % select_chi2)
pl.barh(indices, score, .2, label="score", color='r')
pl.barh(indices + .3, training_time, .2, label="training time", color='g')
pl.barh(indices + .6, test_time, .2, label="test time", color='b')
pl.yticks(())
pl.legend(loc='best')
pl.subplots_adjust(left=.25)
pl.subplots_adjust(top=.95)
pl.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    pl.text(-.3, i, c)
if save_fig:
    pl.savefig("figures/benchmark_%d_feat" % select_chi2)
pl.show()

# <codecell>






# <codecell>


