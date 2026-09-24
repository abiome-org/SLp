"""Inside the legacy container: PiLSL node-feature dict. Feng's release feeds random 600-d vectors
(pilsl_random_feature.npy) and, with add_feat_emb=False, never uses them in the score; we do the same (seed 0)."""
import numpy as np, pandas as pd
N = len(pd.read_csv('/work/data/preprocessed_data/meta_table_9845.csv'))
rng = np.random.RandomState(0)
np.save('/work/data/preprocessed_data/pilsl_data/pilsl_random_feature.npy', {i: rng.randn(600) for i in range(N + 1)})
print('features for', N + 1)
