import json
import re
import iopro.pyodbc as pyodbc
import pandas as pd
from pandas import Series
from pandas import DataFrame as DF
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from fuzzywuzzy import fuzz
import utils

import logging
from optparse import OptionParser
import sys
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

config_file = "conn_deets.json"
select1 = """
SELECT DISTINCT
       phys_id,
       physicians.first_name,
       physicians.last_name,
       unanglicized_name,
       ordinal,
       pmid,
       article_title as title,
       article_abstract as abstract,
       year,
       affiliation as aff_nullable,
       COALESCE(affiliation, "") as affiliation,
       number_of_authors,
       COALESCE(cit_count, 0) as cit_count,
       up_id
FROM   physicians
       INNER JOIN publication_physician_author_rltn USING (phys_id)
       INNER JOIN publications USING (pmid)
       INNER JOIN unnormalized_physicians USING (up_id)
       INNER JOIN (SELECT phys_id,
                          Count(*) AS match_count
                   FROM   publication_physician_author_rltn
                   GROUP  BY phys_id) matches USING (phys_id)
       LEFT JOIN (SELECT cited_pmid AS pmid,
                         Count(*)   AS cit_count
                  FROM   publication_citations
                  GROUP  BY cited_pmid) cits USING (pmid)
       INNER JOIN organization_publication_physician_author_rltn USING (ppa_id,
       pmid)
       INNER JOIN found_organizations USING (found_id)
WHERE  match_count > 1
       AND number_of_authors != 0
       AND country = 'USA' 
       AND article_abstract is not null
       AND article_title is not null
       AND is_book = 0
       AND year > 1990
       AND physicians.first_name = unnormalized_physicians.first_name
"""

ab_ti_query="""
SELECT pmid, article_abstract, article_title 
FROM   publications
WHERE  article_abstract is not null
       AND article_title is not null
       AND is_book = 0
       AND year > 1990"""

id = "phys_id"
up = "up_id"
f = "first_name"
l = "last_name"
un = "unanglicized_name"
ord = "ordinal"
pmid = "pmid"
ti = "title"
ab = "abstract"
y = "year"
aff = "affiliation"
affn = "aff_nullable"
numa = "number_of_authors"
coa = "coauthors"
cit = "cit_count"

s = json.load(open(config_file))
conn_str = ('driver=%s; server=%s; uid=%s; pwd=%s; db=%s' %
            (s['driver'], s['server'], s['uid'], s['pwd'], s['db']))
conn = pyodbc.connect(conn_str)
"""
make the pairs
"""
df = pd.io.sql.read_sql(select1, conn)
df[ab] = df[ab].map(lambda s : " ".join([utils.preprocess(x) for x in json.loads(s).itervalues()]))
df[ti] = df[ti].map(lambda x: utils.preprocess(x))
# see
# http://stackoverflow.com/questions/13446480/python-pandas-remove-entries-based-on-the-number-of-occurrences#comment18556837_13447176
# for a better way?
counts = df.groupby(un).size()
counts = counts[counts != 1]
df = df[df[un].isin(counts.index.values)]
cursor = conn.cursor()
df[coa] = df.apply(lambda x: utils.query_coauths(cursor, int(x[pmid]), int(x[id])), axis=1)['pmid']
cursor.close()


ungroup = df.groupby(un)
bases_idx = []
matches_idx = []
for group in ungroup:
    utils.get_sample_idx(group, bases_idx, matches_idx)
bases = df.ix[bases_idx]
pairs_a = pd.concat([bases, utils.sample_df(bases, bases.shape[0]*3)], axis=0, ignore_index=True)
matches = df.ix[matches_idx]
others = df.drop(matches_idx)
others = df.drop(bases_idx)
pairs_b = pd.concat([matches, utils.sample_df(others, bases.shape[0]*3)], axis=0, ignore_index=True)
pairs_a.columns = [c + "_a" for c in pairs_a.columns.values]
pairs_b.columns = [c + "_b" for c in pairs_b.columns.values]
pairs = pd.concat([pairs_a, pairs_b], axis = 1)
"""
get the tfs
"""
pmids = [x for x in df[pmid]]
abstracts = []
titles = []
cursor = conn.cursor()
for row in cursor.execute(ab_ti_query).fetchall():
    if int(row[0]) not in pmids:
        stract = " ".join([utils.preprocess(x) for x in json.loads(row[1]).itervalues()])
        abstracts.append(stract)
        titles.append(utils.preprocess(row[2]))

conn.close()
abstracts_10k = np.random.choice(abstracts, 10000, replace=False)
tfidf_ab = TfidfVectorizer(analyzer='char', ngram_range=(3,5), sublinear_tf=True, max_features=50000, max_df=0.5, stop_words='english')
# tfidf_ab5 = TfidfVectorizer(analyzer='char', ngram_range=(5,5), sublinear_tf=True, max_features=200, max_df=0.5, stop_words='english')
# tfidf_ab3 = TfidfVectorizer(analyzer='char', ngram_range=(3,3), sublinear_tf=True, max_features=200, max_df=0.5, stop_words='english')
tfidf_ti = TfidfVectorizer(analyzer='word', sublinear_tf=True, max_df=0.5, stop_words='english')
tfidf_ab.fit(abstracts_10k)
# tfidf_ab3.fit(abstracts_10k)
# tfidf_ab5.fit(abstracts_10k)
tfidf_ti.fit(titles)

trainRows = np.random.choice(pairs.index.values, int(np.floor(3*pairs.shape[0] / 4)), replace=False)
pairsTrain = pairs.ix[trainRows]
ab_diffs_train = np.abs(tfidf_ab.transform(pairsTrain[ab+"_a"]) - tfidf_ab.transform(pairsTrain[ab+"_b"]))
ti_diffs_train = np.abs(tfidf_ti.transform(pairsTrain[ti+"_a"]) - tfidf_ti.transform(pairsTrain[ti+"_b"]))
# ab3_diffs_train = tfidf_ab3.transform(pairsTrain[ab+"_a"]) - tfidf_ab5.transform(pairsTrain[ab+"_b"])
# ab5_diffs_train = tfidf_ab5.transform(pairsTrain[ab+"_a"]) - tfidf_ab5.transform(pairsTrain[ab+"_b"])
pairsTest = pairs.drop(trainRows)
ab_diffs_test = np.abs(tfidf_ab.transform(pairsTest[ab+"_a"]) - tfidf_ab.transform(pairsTest[ab+"_b"]))
ti_diffs_test = np.abs(tfidf_ti.transform(pairsTest[ti+"_a"]) - tfidf_ti.transform(pairsTest[ti+"_b"]))
# ab3_diffs_test = tfidf_ab3.transform(pairsTest[ab+"_a"]) - tfidf_ab5.transform(pairsTest[ab+"_b"])
# ab5_diffs_test = tfidf_ab5.transform(pairsTest[ab+"_a"]) - tfidf_ab5.transform(pairsTest[ab+"_b"])
X_train_df = pairsTrain.apply(utils.compare, axis=1)
X_train_df = X_train_df - X_train_df.mean()
X_train_df = X_train_df / X_train_df.std()
y_train_df = (pairsTrain[un+"_a"] == pairsTrain[un+"_b"])
X_test_df = pairsTest.apply(utils.compare, axis=1)
X_test_df = X_test_df - X_test_df.mean()
X_test_df = X_test_df / X_test_df.std()
y_test_df = (pairsTest[un+"_a"] == pairsTest[un+"_b"])

X_train_comps = csr_matrix(X_train_df)
X_train_tfidf = hstack([ab_diffs_train, ti_diffs_train])
X_train = hstack([X_train_comps, X_train_tfidf])
y_train = np.array(y_train_df)

X_test_comps = csr_matrix(X_test_df)
X_test_tfidf = hstack([ab_diffs_test, ti_diffs_test])
X_test = hstack([X_test_comps, X_test_tfidf])
y_test = np.array(y_test_df)

X_all = vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

utils.save_csr("X_train_comps", X_train_comps)
utils.save_coo("X_train_tfidf", X_train_tfidf)
utils.save_coo("X_train", X_train)
utils.save_array("y_train", y_train)

utils.save_csr("X_test_comps", X_test_comps)
utils.save_coo("X_test_tfidf", X_test_tfidf)
utils.save_coo("X_test", X_test)
utils.save_array("y_test", y_test)

utils.save_coo("X_all", X_all)
utils.save_array("y_all", y_all)

feature_names = [x for x in X_train_df.columns.values]
[feature_names.append(x) for x in tfidf_ab.get_feature_names()]
[feature_names.append(x) for x in tfidf_ti.get_feature_names()]
feature_names = Series(feature_names)
feature_names.to_pickle("feature_names.pickle")
