"""Runs inside the legacy container (py3.7): writes the Feng indep_test split pickle and id dicts."""
import pickle, numpy as np, scipy.sparse as sp
z = np.load('/work/data/data_split/slb_split_arrays.npz')
N = int(z['N'][0])
def g(p):
    m = sp.csr_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(N, N))
    return m + m.T
def obj(x):
    a = np.empty(1, dtype=object); a[0] = x; return a
P = [z['fit_pos'], z['val_pos']]; Q = [z['fit_neg'], z['val_neg']]
import os
if os.environ.get('SLB_BALANCE') == '1':  # original KG4SL/SLGNN protocol: 1:1 negatives (here: measured negatives, subsampled)
    rng = np.random.RandomState(0); Q[0] = Q[0][rng.choice(len(Q[0]), size=min(len(Q[0]), len(P[0])), replace=False)]
pos = [obj(g(P[0])), [g(P[1])], [g(P[1])], obj(P[0]), [P[1]], [P[1]]]
neg = [obj(g(Q[0])), [g(Q[1])], [g(Q[1])], obj(Q[0]), [Q[1]], [Q[1]]]
with open('/work/data/data_split/CV1_50_slb%s.pkl' % ('bal' if os.environ.get('SLB_BALANCE') == '1' else ''), 'wb') as f:
    pickle.dump([pos, neg], f)
d = {i: i for i in range(N)}
np.save('/work/data/preprocessed_data/all_uni_id_cut.npy', d)
np.save('/work/data/preprocessed_data/wo_compt/all_uni_id_cut.npy', d)
print('pkl written', N, [len(x) for x in P + Q])
