from pandas import DataFrame as DF
from scipy.sparse import hstack
from scipy.sparse import vstack
from methods import get_data
from pprint import pprint
from sklearn import linear_model, decomposition, datasets, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
import numpy as np
import logging
from time import time
import pickle

# constants
dataset_version = "data_aml_char_1000_short_ti"
print_topX=10
print_report=True
print_cm=False
save_fig=True
show_fig=False
use_comps=False
use_ab=True
use_ti=True
test_percent=.25

(feature_names_comps, feature_names_ab,
     feature_names_ti, X_comps, X_ab, X_ti, y) = get_data(dataset_version)


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
X_train = []
X_test = []
if use_comps:
    feature_names.extend(feature_names_comps)
    X_train.append(X_c_train)
    X_test.append(X_c_test)
if use_ab:
    feature_names.extend(feature_names_ab)
    X_train.append(X_ab_train)
    X_test.append(X_ab_test)
if use_ti:
    feature_names.extend(feature_names_ti)
    X_train.append(X_ti_train)
    X_test.append(X_ti_test)

feature_names = np.array(feature_names)
X_train = hstack(X_train)
X_test = hstack(X_test)
print("done assembling features in %fs" % (time() - t0))

# <codecell>

# Models we will use
X_train = X_train.todense()
X_test = X_test.todense()


fitted=[]

def grid_search(estimator):
    # try:
    pipeline = Pipeline([(estimator[0], estimator[1])])
    param_grid = dict(estimator[2].items())
    print()
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(param_grid)
    clf = GridSearchCV(pipeline, param_grid, n_jobs=3, verbose=1)
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()

    clf.fit(X_train_t, y_train)
    train_time = time() - t0
    pickle.dump(clf, open("_".join([name for name, _ in pipeline.steps]), 'wb'))
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
    score = f1_score(y_test, y_pred)
    fitted.append([score, clf])
    print("f1-score:   %0.3f" % score)
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print()
    # except Exception, e:
    #     print()
    #     print("="*80)
    #     print(e)
    #     print("="*80)
    #     print()
    #     print()

rbm_grd = [dict(n_iter=35, n_components=100, learning_rate=1),
           dict(n_iter=35, n_components=200, learning_rate=1),
           dict(n_iter=100, n_components=100, learning_rate=1)]

pca_grd = [dict(n_components=1),
           dict(n_components=5),
           dict(n_components=10),
           dict(n_components=50)]

sp_grd = [dict(percentile=10),
          dict(percentile=25),
          dict(percentile=50)]

svcft_grd = [dict(C=.5),
             dict(C=1.0),
             dict(C=10)]

rf_grd = {
    'rf__max_features':['sqrt','log2'],
    'rf__n_estimators':[50,100,200,500]
    }

abc_grd = {
    'abc__n_estimators':[25, 100]
    }

svcl_grd = {
    'svcl__C':[.01,.1,1,10,100,1000]
    }

svcrbf_grd = {
    'svcrbf__C':[1,10,100,1000],
    'svcrbf__gamma':[0.001,0.0001,0.01]
    }

lr_grd = {
    'lr__C':[1,10,100,1000]
    }

transformers = [('pca', PCA(), pca_grd),
                ('sp', SelectPercentile(f_classif), sp_grd),
                ('svcft', LinearSVC(penalty="l1", dual=False), svcft_grd),
                ('rbm', BernoulliRBM(random_state=0), rbm_grd)
                ] # l1 leads to sparsity

estimators = [('rf', RandomForestClassifier(), rf_grd),
              ('svcl', SVC(kernel='linear'), svcl_grd),
              ('abc', AdaBoostClassifier(), abc_grd),
              ('svcrbf', SVC(kernel='rbf'), svcrbf_grd),
              ('lr', LogisticRegression(), lr_grd)]

for transformer in transformers:
    for params in transformer[2]:

        clf = transformer[1]
        print("Transforming using " + str(type(clf)) + ' with ' + " ".join([x+'-'+str(y) for x,y in params.items()]))
        
        t0 = time()
        clf.set_params(**params)
        X_train_t = clf.fit_transform(X_train, y_train)
        X_test_t = clf.transform(X_test)
        train_time = time() - t0
        pickle.dump(clf, open(dataset_version+"/"+transformer[0]+"_"+"_".join([x+'-'+str(y) for x,y in params.items()]), 'wb'))
        print("train time: %0.3fs" % train_time)
        
        for estimator in estimators:
            grid_search(estimator)
