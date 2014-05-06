import os
import json
import re
import iopro.pyodbc as pyodbc
import pandas as pd
from pandas import Series
from pandas import DataFrame as DF
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
import utils
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import sklearn
import scipy

#("char", (3,5), "aml_char")

deets=[("char", (3,5), "aml_char_1000_short_ti")]

for deet in deets:

    dataset_version="data_%s" % deet[2]
    ab_analyzer=deet[0]
    ab_ngram_range=deet[1]
    max_abstracts=20000
    ab_max_feat=1000
    dataset_description="%d abstracts, %d max features, %s analyzer, range %s" % (max_abstracts, ab_max_feat, ab_analyzer, ab_ngram_range)
    additional_info="acute megacocaryoblastic leukemia"
    max_df = .5

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
           INNER JOIN organization_publication_physician_author_rltn USING (ppa_id, pmid)
           INNER JOIN found_organizations USING (found_id)
    WHERE  match_count > 1
           AND number_of_authors != 0
           AND country = 'USA' 
           AND article_abstract is not null
           AND article_title is not null
           AND is_book = 0
           AND year > 1990
           AND physicians.first_name = unnormalized_physicians.first_name
           AND cond_id = 47
    """

    ab_ti_query="""
    SELECT pmid, article_abstract, article_title 
    FROM   publications
    INNER JOIN publication_condition_rltn
    USING (pmid)
    WHERE  article_abstract is not null
           AND article_title is not null
           AND is_book = 0
           AND year > 1990
           AND pmid % 2 = 0
           AND cond_id = 47"""

    select2 = """
    select phys_id, first_name as ph_first, last_name as ph_last, cond_id from physicians inner join 
        ( select phys_id, up_id from publication_p
    """


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

    """
    save the world
    """
    print "%d abstracts" % len(abstracts)
    if len(abstracts) > max_abstracts:
        abstracts = np.random.choice(abstracts, max_abstracts, replace=False)
    print "now %d abstracts" % len(abstracts)

    tfidf_ab = TfidfVectorizer(analyzer=ab_analyzer, ngram_range=ab_ngram_range, sublinear_tf=True, max_features=ab_max_feat, max_df=max_df, stop_words='english')
    # tfidf_ab5 = TfidfVectorizer(analyzer='char', ngram_range=(5,5), sublinear_tf=True, max_features=200, max_df=0.5, stop_words='english')
    # tfidf_ab3 = TfidfVectorizer(analyzer='char', ngram_range=(3,3), sublinear_tf=True, max_features=200, max_df=0.5, stop_words='english')
    tfidf_ti = TfidfVectorizer(analyzer='word', ngram_range=(1,2), sublinear_tf=True, max_df=max_df, max_features=ab_max_feat, stop_words='english')
    tfidf_ab.fit(abstracts)
    # tfidf_ab3.fit(abstracts_10k)
    # tfidf_ab5.fit(abstracts_10k)
    tfidf_ti.fit(titles)


    ab_diffs = np.abs(tfidf_ab.transform(pairs[ab+"_a"]) - tfidf_ab.transform(pairs[ab+"_b"]))
    ti_diffs = np.abs(tfidf_ti.transform(pairs[ti+"_a"]) - tfidf_ti.transform(pairs[ti+"_b"]))
    comps_diffs_df = pairs.apply(utils.compare, axis=1)
    comps_diffs = comps_diffs_df - comps_diffs_df.mean()
    comps_diffs = np.array(comps_diffs / comps_diffs.std())

    y = np.array(pairs[un+"_a"] == pairs[un+"_b"])

    feature_names_comps = np.array(comps_diffs_df.columns.values)
    feature_names_ab = np.array(tfidf_ab.get_feature_names())
    feature_names_ti = np.array(tfidf_ti.get_feature_names())

    if not os.path.exists(dataset_version):
        os.makedirs(dataset_version)

    utils.save_csr("%s/X_ab" % dataset_version, ab_diffs)
    utils.save_csr("%s/X_ti" % dataset_version, ti_diffs)
    utils.save_array("%s/X_comps" % dataset_version, comps_diffs)
    utils.save_array("%s/y" % dataset_version, y)

    utils.save_array("%s/feature_names_ab" % dataset_version, feature_names_ab)
    utils.save_array("%s/feature_names_ti" % dataset_version, feature_names_ti)
    utils.save_array("%s/feature_names_comps" % dataset_version, feature_names_comps)
    pairs[['pmid_a', 'pmid_b', un+'_a', un+'_b']].to_pickle("%s/feature_names_comps" % dataset_version)


    text_file = open("%s/description.txt" % dataset_version, "w")
    text_file.write(dataset_description)
    text_file.close()
    print("done with %s" % dataset_version)
