from pathlib import Path
import argparse, hashlib, json
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"


def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def build(perturb_name="perturbseq_world_v2",output_name="perturbseq_world_v3",update_basal=True):
    dep_path=OUT/"depmap_world.npz"; perturb_path=OUT/f"{perturb_name}.npz"; dep=np.load(dep_path); model=pd.read_csv(ROOT/"data/depmap24q2/Model.csv"); ids={x:i for i,x in enumerate(dep["model_ids"].astype(str))}; rpe=[x for x in model.loc[model.CellLineName.astype(str).str.startswith("RPE1-"),"ModelID"].astype(str) if x in ids]; mapping={"adamson2016":["ACH-000551"],"dixit2016":["ACH-000551"],"norman2019":["ACH-000551"],"replogle2022_k562":["ACH-000551"],"replogle2022_rpe1":rpe,"wessels2023":["ACH-000146"],"joung2023":[],"dixit_bmdc_3h":[],"dixit_bmdc_0h":[],"dixit_k562_high_moi":["ACH-000551"]}; perturb=np.load(perturb_path); sources=perturb["sources"].astype(str); state=np.asarray([dep["cell_state"][[ids[x] for x in mapping[name]]].mean(0) if mapping[name] else np.zeros(dep["cell_state"].shape[1]) for name in sources],"float32"); known=np.asarray([bool(mapping[x]) for x in sources])
    if update_basal: np.savez_compressed(OUT/"basal_context.npz",**{k:dep[k] for k in dep.files},source_state=state,sources=sources,source_known=known)
    np.savez_compressed(OUT/f"{output_name}.npz",**{k:perturb[k] for k in perturb.files},context_state=state,context_known=known)
    audit={"schema":"sl-predict-basal-context-v1","method":"checksum-derived DepMap 24Q2 expression/dependency state; exact K-562 and THP-1, mean of five available RPE1 engineered derivatives, explicit zero unknown state for hESC","dimensions":state.shape[1],"perturbation_state_dimensions":int(perturb["state_dimensions"]) if "state_dimensions" in perturb.files else 32,"depmap_cells":len(dep["model_ids"]),"perturbation_input_sha256":sha(perturb_path),"depmap_input_sha256":sha(dep_path),"sources":[{"source":x,"model_ids":mapping[x],"known":bool(k)} for x,k in zip(sources,known)],"dependency_auxiliary_training_genes":int(dep["train_gene"].sum())}; (OUT/f"{output_name}.json").write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--perturb-name",default="perturbseq_world_v2"); p.add_argument("--output-name",default="perturbseq_world_v3"); p.add_argument("--skip-basal",action="store_true"); a=p.parse_args(); build(a.perturb_name,a.output_name,not a.skip_basal)
