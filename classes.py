import utils
from sklearn.metrics import classification_report
import numpy as np
from sklearn.utils.extmath import density
import logging
from time import time
from sklearn import metrics
from sklearn.svm import LinearSVC

class L1LinearSVC(LinearSVC):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

class Benchmarker:
    def __init__(self, X_train, y_train, X_test, y_test, fn,  topX, rep, cm):
        self.print_topX=topX
        self.print_report=rep
        self.print_cm=cm
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.feature_names=fn
        
    def benchmark(self, clf):
        print_topX=self.print_topX
        print_report=self.print_report
        print_cm=self.print_cm
        X_train=self.X_train
        y_train=self.y_train
        X_test=self.X_test
        y_test=self.y_test
        feature_names=self.feature_names
        categories = ["1"]
        
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
                print("top 10 keywords per class:")
                for i, category in enumerate(categories):
                    topX = np.argsort(clf.coef_[i])[-print_topX:]
                    print(trim("%s: %s"
                               % (category, " ".join(feature_names[topX]))))
            print()

        if print_report:
            print("classification report:")
            print(classification_report(y_test, pred))
                                            # target_names=categories))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time, clf, pred
