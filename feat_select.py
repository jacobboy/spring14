# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
from pandas import Series
from pandas import DataFrame as DF
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
import utils
from utils import trim
from methods import get_data
from classes import L1LinearSVC
from classes import Benchmarker

import logging
from time import time
import numpy as np
import sklearn
import scipy
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.utils.extmath import density
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
select_chi2s=[10,100,500,1000]
print_topX=10
print_report=True
print_cm=False
save_fig=True
show_fig=False
use_comps=True
use_ab=False
use_ti=True
test_percent=.25
categories=["1"]

# <codecell>

###############################################################################
# Benchmark method
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if print_topX:
            print("top feature per class:")
            for i, category in enumerate(categories):
                # topX = np.min(clf.coef_.shape[1], print_topX)
                topX = np.argsort(clf.coef_[i])[-print_topX:][::-1]
                print(trim("%s: %s"
                           % (category, " | ".join(feature_names[topX]))))
        print()

    if print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred))
                                            # target_names=categories))

    if print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time, clf, pred

# <codecell>

################################################################################
# load data

(feature_names_comps, feature_names_ab,
 feature_names_ti, X_comps, X_ab, X_ti, y) = get_data(dataset_version)

# <codecell>

for select_chi2 in select_chi2s:

# <codecell>

    ############################################################################
    # extract features via chi2
    # assemble X_train, X_test, and feature_names
    print("assembling features")
    t0 = time()

    (X_c_train, X_c_test, X_ab_train, X_ab_test, X_ti_train, X_ti_test,
     y_train, y_test) = train_test_split(X_comps, X_ab, X_ti, y,
                                         test_size=test_percent)

    feature_names = []
    feature_names_tfidf = []
    X_train = []
    X_train_tfidf = []
    X_test = []
    X_test_tfidf = []
    if use_comps:
        feature_names.extend(feature_names_comps)
        X_train.append(X_c_train)
        X_test.append(X_c_test)
    if use_ab:
        feature_names_tfidf.extend(feature_names_ab)
        X_train_tfidf.append(X_ab_train)
        X_test_tfidf.append(X_ab_test)
    if use_ti:
        feature_names_tfidf.extend(feature_names_ti)
        X_train_tfidf.append(X_ti_train)
        X_test_tfidf.append(X_ti_test)
    
    if X_train_tfidf:
        print("Extracting %d best features by a chi-squared test" % select_chi2)
        t1 = time()

        feature_names_tfidf = np.array(feature_names_tfidf)
        X_train_tfidf = hstack(X_train_tfidf)
        X_test_tfidf = hstack(X_test_tfidf)

        ch2 = SelectKBest(chi2, k=select_chi2)
        ch2.fit(X_train_tfidf, y_train)
        
        feature_names_ch2 = feature_names_tfidf[ch2.get_support()]
        X_train_ch2 = ch2.transform(X_train_tfidf)
        X_test_ch2 = ch2.transform(X_test_tfidf)

        feature_names.extend(feature_names_ch2)
        X_train.append(X_train_ch2)
        X_test.append(X_test_ch2)
        
        print("done in %fs" % (time() - t1))

    feature_names = np.array(feature_names)
    if use_ab or use_ti:
        X_train = hstack(X_train)
        X_test = hstack(X_test)
    else:
        X_train = X_c_train
        X_test = X_c_test
    print("done assembling features in %fs" % (time() - t0))

# <codecell>

    # Finally, get to work
    results = []

# <codecell>

    # for clf, name in (
    #         (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    #         (Perceptron(n_iter=50), "Perceptron"),
    #         (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
    #         (KNeighborsClassifier(n_neighbors=10), "kNN")):
    #     print('=' * 80)
    #     print(name)
    #     results.append(benchmark(clf))

# <codecell>

    
    # Train Liblinear models
    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3)))

# <codecell>

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    results.append(benchmark(L1LinearSVC()))

# <codecell>

    # Train SGD models
    for penalty in ["l2", "l1", "elasticnet"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
    
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

# <codecell>

    # # Train NearestCentroid without threshold
    # print('=' * 80)
    # print("NearestCentroid (aka Rocchio classifier)")
    # results.append(benchmark(NearestCentroid()))

# <codecell>

    # Input X must be non-negative
    # # Train sparse Naive Bayes classifiers
    # print('=' * 80)
    # print("Naive Bayes")
    # results.append(benchmark(MultinomialNB(alpha=.01)))
    # results.append(benchmark(BernoulliNB(alpha=.01)))

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
        if use_comps:
            namemod = "_" + str(select_chi2) + "_comps"
        else:
            namemod = "_" + str(select_chi2) + "_nocomps"
        pl.savefig("figures/benchmark%s" % namemod)
    if show_fig:
        pl.show()
    pl.close()

# <codecell>






# <codecell>


