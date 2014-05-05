import json
import re
import iopro.pyodbc as pyodbc
import pandas as pd
from pandas import Series
from pandas import DataFrame as DF
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from unicodedata import normalize as unorm
import scipy

ida = "phys_id" + "_a"
upa = "up_id" + "_a"
fa = "first_name" + "_a"
la = "last_name" + "_a"
una = "unanglicized_name" + "_a"
orda = "ordinal" + "_a"
pmida = "pmid" + "_a"
tia = "title" + "_a"
aba = "abstract" + "_a"
ya = "year" + "_a"
affa = "affiliation" + "_a"
affna = "aff_nullable" + "_a"
numaa = "number_of_authors" + "_a"
coaa = "coauthors" + "_a"
cita = "cit_count" + "_a"

idb = "phys_id" + "_b"
upb = "up_id" + "_b"
fb = "first_name" + "_b"
lb = "last_name" + "_b"
unb = "unanglicized_name" + "_b"
ordb = "ordinal" + "_b"
pmidb = "pmid" + "_b"
tib = "title" + "_b"
abb = "abstract" + "_b"
yb = "year" + "_b"
affb = "affiliation" + "_b"
affnb = "aff_nullable" + "_b"
numab = "number_of_authors" + "_b"
coab = "coauthors" + "_b"
citb = "cit_count" + "_b"

def count_missing(frame):
    return (frame.shape[0] * frame.shape[1]) - frame.count().sum()

def un_utf8(s):
    return unorm('NFKD', s).encode('ascii','ignore')

coauth_query="""SELECT last_name from physicians inner join publication_physician_author_rltn using (phys_id) where pmid = ? and phys_id != ?"""
def query_coauths(cursor, pmid, phys_id):
    x = [row[0] for row in cursor.execute(coauth_query, pmid, phys_id).fetchall()]
    # if len(x) == 0:
    #     print "empty"
    # else:
    #     print ", ".join(x)
    return [x]

def get_sample_idx(df, bases, matches):
    choice = np.random.choice(df[1].index.values, 2, replace=False)
    bases.append(choice[0])
    matches.append(choice[1])

def sample_df(df, num):
    rows = np.random.choice(df.index, num)
    return df.ix[rows]

def preprocess(string):
    x = " ".join([text for text in string.split() if not any([c.isdigit() for c in text])])
    return re.sub(ur"\p{P}+", "", x.lower())

# label_indices_no_digits = [i for i, v in enumerate(tfidf_ngram.get_feature_names()) if not any(x.isdigit() for x in v)]
# label_indices_no_digits = [i for i, v in enumerate(tfidf_ngram.get_feature_names()) if re.match('^[a-zA-Z\s]+$', v)]

def compare(s):
    from numpy import abs
    from fuzzywuzzy import fuzz
    tisim = 'title_sim'
    affsim = 'aff_sim'
    cpd = 'cit_peryear_diff'
    yd = 'year_diff'
    od = 'ord_diff'
    rod = 'rel_ord_diff'
    ncd = 'num_coauth_diff'
    scc = 'same_coauth_count'
    x = Series(
        {
            tisim: fuzz.token_set_ratio(s[tia], s[tib]),
            affsim: fuzz.token_set_ratio(s[affna], s[affnb]),
            cpd: abs((np.float(s[cita])/(2015 - s[ya])) - (np.float(s[citb])/(2015 - s[yb]))),
            yd: abs(s[ya] - s[yb]),
            od: abs(s[orda] - s[ordb]),
            rod: abs(((np.float(s[numaa]) - s[orda])/s[numaa]) - ((np.float(s[numab]) - s[ordb]) / s[numab])),
            ncd: abs(s[numaa] - s[numab]),
            scc: len([x for x in s[coaa] if x in s[coab]])
        }
    )
    # x['title_sim'] = fuzz.token_set_ratio(s[tia], s[tib])
    # x['aff_sim'] = fuzz.token_set_ratio(s[affna], s[affnb])
    # x['cit_peryear_diff'] = np.abs((s[cita]/(2015 - s[ya])) - (s[citb]/(2015 - s[yb])))
    # x['year_diff'] = abs(s[ya] - s[yb])
    # x['ord_diff'] = abs(s[orda] - s[ordb])
    # x['rel_ord_diff'] = abs(((s[numaa] - s[orda])/s[numaa]) - ((s[numab] - s[ordb]) / s[numab]))
    # x['num_coauth_diff'] = abs(s[numaa] - s[numab])
    # x['same_coauth_count'] = len([x for x in s[coaa] if x in s[coab]])
    return x

def save_csr(name, csr):
    if type(csr) != scipy.sparse.csr_matrix:
        raise Exception("Not a csr_matrix, is " + str(type(csr)))
    from scipy.sparse import csr_matrix
    shape = np.array([x for x in csr.shape])
    np.savez(name, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=shape)

def load_csr(name):
    from scipy.sparse import csr_matrix
    npz = np.load(name+".npz")
    return csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=(npz['shape'][0], npz['shape'][1]))

def save_coo(name, coo):
    if type(coo) != scipy.sparse.coo.coo_matrix:
        raise Exception("Not a coo_matrix, is " + str(type(coo)))
    from scipy.sparse import coo_matrix
    save_csr(name, coo.tocsr())

def load_coo(name):
    from scipy.sparse import coo_matrix
    csr = load_csr(name)
    return csr.tocoo()

def save_array(name, array):
    np.save(name, array)

def load_array(name):
    return np.load(name+".npy")

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    """Balls to that, actually"""
    return s if len(s) <= 320 else s[:317] + "..."
