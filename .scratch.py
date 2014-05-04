

# print("Performing dimensionality reduction using LSA")
# t0 = time()
# lsa = TruncatedSVD(opts.n_components)
# X = lsa.fit_transform(X_train)
#     # Vectorizer results are normalized, which makes KMeans behave as
#     # spherical k-means for better results. Since LSA/SVD results are
#     # not normalized, we have to redo the normalization.
# X = Normalizer(copy=False).fit_transform(X)

# print("done in %fs" % (time() - t0))
# print()


# from sklearn.svm import SVC
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.feature_selection import RFECV
# # Create the RFE object and compute a cross-validated score.
# svc = SVC(kernel="linear")
# rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(Y_all, 2),
#               scoring='accuracy')
# rfecv.fit(X_all, Y_all)













select = SelectPercentile(score_func=chi2, percentile=18)
clf = LogisticRegression(tol=1e-8, penalty='l2', C=7)
countvect_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", binary=False)
badwords = BadWordCounter()
ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char), ])
char_model = Pipeline([('vect', ft), ('select', select), ('logr', clf)])



select = SelectPercentile(score_func=chi2, percentile=16)
 
clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
countvect_char = TfidfVectorizer(ngram_range=(1, 5), analyzer="char", binary=False)
countvect_word = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
badwords = BadWordCounter()
 
ft = FeatureStacker([("badwords", badwords), ("chars", countvect_char),("words", countvect_word)])
char_word_model = Pipeline([('vect', ft), ('select', select), ('logr', clf)])
