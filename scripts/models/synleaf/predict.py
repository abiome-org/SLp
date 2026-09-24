"""Score SLB pairs with a trained SynLeaF checkpoint (run from external/models/synleaf/src with its venv).

usage: python predict.py <ct> <result_dir> <split> <out.csv>
  <result_dir>: ../result/<folder> of a train.py run (hyper_parameters.json + checkpoint.pth); task type is read
  from its hyper_parameters.json (umt = the full SynLeaF model; only_omics / only_kg = stage-1 teachers).
Pairs are scored in both gene orders and averaged (the classifier concatenates gene1|gene2, so it is not
symmetric). Rows whose genes are outside the model gene set get NaN.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path.cwd()))
from accelerate import Accelerator  # noqa: E402

from dataset import MultiModalDataset  # noqa: E402
from model import UMTModel  # noqa: E402

ct, rdir, split, out = sys.argv[1:5]
hp = json.load(open(Path(rdir) / "hyper_parameters.json"))
task = hp["task_type"]
acc = Accelerator()
dev = acc.device
base = f"../data/{ct}"
mm = MultiModalDataset(acc, gene_path=f"{base}/gene.pt",
                       omics_path_dict={o: f"{base}/{o}.npy" for o in hp["omics_types"]},
                       kg_graph_path=f"{base}/kg.pt")
model = UMTModel(
    omics_encoder_params={"omics_count": len(hp["omics_types"]), "omics_input_dim": mm.get_omics_input_dim(),
                          "vae_hidden_dims": [int(x) for x in hp["vae_hidden_dims"]], "vae_dropout": hp["vae_dropout"],
                          "hid_dim": hp["hid_dim"]},
    kg_encoder_params={"entity_vocab_size": mm.get_entity_vocab_size(), "hid_dim": hp["hid_dim"],
                       "in_channels": hp["in_channels"], "hidden_channels": hp["hidden_channels"],
                       "gcn_layers": hp["gcn_layers"], "graph_dropout": hp["graph_dropout"],
                       "num_relations": mm.get_kg_relations_count()},
    gene_final_dim=hp["gene_final_dim"], final_mlp_dropout=hp["final_mlp_dropout"])
ck = torch.load(Path(rdir) / "checkpoint.pth", map_location="cpu", weights_only=True)
UMTModel.load_state_dicts(model, ck["model"])
model.to(dev).eval()
kg = mm.get_kg_graph()

ev = pd.read_parquet(f"{base}/eval_{split}.parquet")
ok = ev.i.notna() & ev.j.notna()
pairs = ev.loc[ok, ["i", "j"]].astype(int).drop_duplicates()
omics = [torch.tensor(mm.omics_data[o], dtype=torch.float32) for o in hp["omics_types"]]
ent = torch.tensor([mm.gene_name_to_entity_idx[mm.gene_idx_to_name[k]] for k in range(len(mm))], dtype=torch.long)


def run(a, b):
    p1 = {"omics_data_list": [x[a].to(dev) for x in omics], "is_training": False}
    p2 = {"omics_data_list": [x[b].to(dev) for x in omics], "is_training": False}
    k1 = {"gene_entity": ent[a].to(dev), "kg_graph": kg}
    k2 = {"gene_entity": ent[b].to(dev), "kg_graph": kg}
    with torch.no_grad():
        r = model(task, p1, p2, k1, k2)
    if task == "ume":
        return 0.5 * (r[0].sigmoid() + r[1].sigmoid()).view(-1)
    logit = r[0] if isinstance(r, tuple) else r
    return logit.sigmoid().view(-1)


scores = []
bs = int(hp["batch_size"])
P = torch.tensor(pairs.values, dtype=torch.long)
for s in range(0, len(P), bs):
    a, b = P[s:s + bs, 0], P[s:s + bs, 1]
    scores.append(0.5 * (run(a, b) + run(b, a)).cpu().numpy())
pairs["score"] = np.concatenate(scores)
ev = ev.merge(pairs, on=["i", "j"], how="left")
ev[["example_id", "score"]].to_csv(out, index=False)
print(f"{task}: scored {ev.score.notna().sum():,}/{len(ev):,} rows ({len(pairs):,} unique pairs)", file=sys.stderr)
