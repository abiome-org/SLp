"""Score a feature CSV with the released De Kegel 2021 RF (sklearn 0.23.1; run inside the slb/dekegel2021 image)."""
import pickle
import sys

import pandas as pd

feat = pd.read_csv("/model/local_data/feature_list.txt").feature.tolist()
rf = pickle.load(open("/model/local_data/results/RF_model.pickle", "rb"))
d = pd.read_csv(sys.argv[1])
m = d.has_features.astype(bool)
d["score"] = float("nan")
d.loc[m, "score"] = rf.predict_proba(d.loc[m, feat].fillna(0).values)[:, 1]
d[["example_id", "score"]].to_csv(sys.argv[2], index=False)
