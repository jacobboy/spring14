import pylab as pl
from pandas import DataFrame as DF
from scipy.sparse import hstack
from scipy.sparse import vstack
from methods import get_data
import utils
from pprint import pprint
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model, decomposition, datasets, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, auc, precision_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import logging
from time import time
import pickle

# constants
dataset_version = "data"
print_topX=10
print_report=True
print_cm=False
save_fig=True
show_fig=False
use_comps=False
use_ab=True
use_ti=True
test_percent=.25
scoring='precision'

(feature_names_comps, feature_names_ab,
     feature_names_ti, X_comps, X_ab, X_ti, y) = get_data(dataset_version)


# <codecell>

############################################################################
# extract features via chi2
# assemble X_train, X_test, and feature_names
print("assembling features")
t0 = time()

fnab = utils.load_array('data/feature_names_ab')
fnti = utils.load_array('data/feature_names_ti')
fn = np.append(fnab, fnti)
X_train = utils.load_coo('data/X_train').todense()
y_train = utils.load_array('data/y_train')
X_test = utils.load_coo('data/X_test').todense()
y_test = utils.load_array('data/y_test')
print("done assembling features in %fs" % (time() - t0))

# <codecell>


fitted=[]

def grid_search(estimator):
    # try:
    # pipeline = Pipeline([(estimator[0], estimator[1])])
    pipeline = estimator[1]
    param_grid = [estimator[2]]
    # print()
    print("Performing grid search for %s %s" % (t_name, e_name))
    # print(str(type(pipeline)))
    # print("parameters:")
    pprint(param_grid)
    clf = GridSearchCV(pipeline, param_grid, n_jobs=4, verbose=0, scoring=scoring)
    # print('_' * 80)
    # print("Training: ")
    # print(clf)
    t0 = time()

    clf.fit(X_train_t, y_train)
    train_time = time() - t0
    pickle.dump(clf, open(dataset_version+ "/" + "".join(t_name.split()) + "_" + "".join(e_name.split()), 'wb'))
    print("train time: %0.3fs" % train_time)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
        print()
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    y_true, y_pred = y_test, clf.predict(X_test_t)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print("f1-score:   %0.3f" % f1)
    print("accuracy:   %0.3f" % acc)
    print("precision:   %0.3f" % prec)
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    true_true = cm[1,1]
    fitted.append((f1, acc, prec, clf, clf.best_estimator_, y_pred, true_true))
    # print()
    # except Exception, e:
    #     print()
    #     print("="*80)
    #     print(e)
    #     print("="*80)
    #     print()
    #     print()

pca_grd = [dict(n_components=3),
           dict(n_components=6),
           dict(n_components=50),
           dict(n_components=100)]

sp_grd = [dict(percentile=10),
          dict(percentile=25),
          dict(percentile=35),
          dict(percentile=50)]

svcft_grd = [dict(C=.5),
             dict(C=.1),
             dict(C=5)]

rbm_grd = [dict(n_iter=35, n_components=100, learning_rate=1),
           dict(n_iter=35, n_components=200, learning_rate=1),
           dict(n_iter=100, n_components=100, learning_rate=1)]

t_names = ['PCA 3', 'PCA 6', 'PCA 50', 'PCA 100', 'SP .1', 'SP .25', 'SP .35', 'SP .5', 'LSVC C .5', 'LSVC C .1', 'LSVC C 5', 'RBM 1', 'RBM 2', 'RBM 3']

rf_grd = {
    'max_features':['sqrt','log2'],
    'n_estimators':[200,500]
    }

abc_grd = {
    'n_estimators':[25,500], # was [25,100]
    'base_estimator':[DecisionTreeClassifier(min_samples_split=2), DecisionTreeClassifier(min_samples_split=20)]
    }

svcl_grd = {
    'C':[.1, .5, 1, 5]
    }

svcrbf_grd = {
    'C':[10,100,1000,2000],
    'gamma':[0.001,0.0001,0.01]
    }

lr_grd = {
    'C':[1,10,20,50]
    }

nb_grd = {
    'alpha':[0.1,0.5,1]
    }

e_names = ['RF', 'SVC L', 'ADA', 'SVC RBF', 'LOGIT', 'NB']

transformers = [('pca', PCA(), pca_grd),
                ('sp', SelectPercentile(f_classif), sp_grd),
                ('svcft', LinearSVC(penalty="l1", dual=False), svcft_grd),
                ('rbm', BernoulliRBM(random_state=0), rbm_grd)
                ] # l1 leads to sparsity

estimators = [('rf', RandomForestClassifier(compute_importances=True), rf_grd),
              ('svcl', SVC(kernel='linear'), svcl_grd),
              ('abc', AdaBoostClassifier(), abc_grd),
              ('svcrbf', SVC(kernel='rbf'), svcrbf_grd),
              ('lr', LogisticRegression(), lr_grd),
              ('nb', BernoulliNB(), nb_grd)]



# clf.fit(X_train, y_train)
# y_true, y_pred = y_test, clf.predict(X_test)
# f1 = f1_score(y_test, y_pred)
# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred)
# print("f1-score:   %0.3f" % f1)
# print("accuracy:   %0.3f" % acc)
# print("precision:   %0.3f" % prec)
# print()
# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# cr = classification_report(y_true, y_pred)
# print(cr)
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# true_true = cm[1,1]



fitted_transformers = []
i = 0
t_name=''
# for transformer in transformers:
#     for params in transformer[2]:
#         # try:
#         if True:
#             t_name = t_names[i]
#             i+=1
#             clf = transformer[1]
#             print("Transforming using " + str(type(clf)) + ' with ' + " ".join([x+'-'+str(y) for x,y in params.items()]))

#             t0 = time()
#             clf.set_params(**params)
#             X_train_t = clf.fit_transform(X_train, y_train)
#             fitted_transformers.append(clf)
#             X_test_t = clf.transform(X_test)
#             train_time = time() - t0
#             pickle.dump(clf, open(dataset_version+"/"+"".join(t_name.split()), 'wb'))
#             print("train time: %0.3fs" % train_time)
#             j = 0
X_train_t = X_train
X_test_t = X_test
j=0
for estimator in estimators:
    e_name = e_names[j]
    j+=1
    grid_search(estimator)
