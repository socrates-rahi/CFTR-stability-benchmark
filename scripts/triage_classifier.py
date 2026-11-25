"""Small cross-validated triage classifier: predicts severe destabilisation
(dTm <= -5 C) of CFTR NBD mutations from FoldX, Rosetta (hrp, lrp) and DDMut scores.
Leave-one-position-out CV. Exploratory; n = 43. Run: python3 triage_classifier.py ../data/benchmark.csv"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d = pd.read_csv(sys.argv[1]).dropna(subset=['ddmut_fwd']).reset_index(drop=True)
d['ddmut'] = -d.ddmut_fwd                      # DDMut: negative = destabilising -> flip
X = d[['foldx','rosetta_hrp','rosetta_lrp','ddmut']].apply(lambda c: np.sign(c)*np.log1p(c.abs())).values
y = (d.exp_dTm <= -5).astype(int).values
key = (d.domain + d.mutation.str[1:-1]).values
oof = np.zeros(len(d))
for k in np.unique(key):
    te = key == k
    m = LogisticRegression(C=0.5, max_iter=1000).fit(X[~te], y[~te])
    oof[te] = m.predict_proba(X[te])[:, 1]
print('Severe destabiliser, LOPO-CV AUC (logistic stack): %.2f' % roc_auc_score(y, oof))
for j, n in enumerate(['foldx','rosetta_hrp','rosetta_lrp','ddmut']):
    print('  single predictor %-12s AUC %.2f' % (n, roc_auc_score(y, X[:, j])))
