import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/sl_predict"; sys.path.insert(0,str(Path(__file__).parent))
from musl import tcga_relation_cv3

def main():
    base=OUT/"native_spectral_safe_intervention_basal_perturbseq_v3"; head32=OUT/"native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3"; state96=OUT/"native_spectral_safe_intervention_basal_perturbseq_state96_p12_d3_t10_r3"; residual=OUT/"native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; path=residual/"musl_cv3_replogle_mutual_induction.json"
    result=tcga_relation_cv3(base/"world_model.pt",head32/"interaction_shrinkage_head.pt",state96/"world_model.pt",state96/"interaction_shrinkage_head.pt",residual/"world_model.pt",residual/"tcga_relation_head.pt",OUT/"features_spectral_safe.npz",ROOT/"data/feng2024/data/preprocessed_data/meta_table_9845.csv",ROOT/"data/models/MuSL/processed_data/meta_table_7684.csv",OUT/"basal_context.npz",path,crossmodal_path=OUT/"replogle_mutual_induction.npz",crossmodal_reliability=.2919864822519491)
    rename={"fixed_crossmodal_fusion":"fixed_mutual_induction_fusion","retained_tcga_fusion":"retained_score","crossmodal_relation":"mutual_induction","crossmodal_coverage":"mutual_induction_coverage"}; result["rows"]=[{rename.get(k,k):v for k,v in row.items()} for row in result["rows"]]; result["mean"]={rename.get(k,k):v for k,v in result["mean"].items()}; result.update({"schema":"sl-predict-replogle-mutual-induction-musl-v1","protocol":"One locked label-free evaluation on both official MuSL CV3 seeds: positive K562/RPE1 mutual partner-expression induction fused with the retained cellular-state/TCGA rank using its independent observed cross-cell Spearman weight; unsupported pairs are neutral; no fitting, sign, weight, source, transform, missing-value, support, fold or seed selection","mutual_induction_reliability_weight":result.pop("crossmodal_reliability_weight")}); path.write_text(json.dumps(result,indent=2)); print(json.dumps({"mean":result["mean"],"advanced":result["advanced"]},indent=2))

if __name__=="__main__": main()
