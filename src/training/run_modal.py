from pathlib import Path
import modal

ROOT = Path.cwd()
LOCAL = (ROOT / "src/training/world_model.py").exists()
MODEL = ROOT / "src/training/world_model.py" if LOCAL else Path("/root/world_model.py")
MODEL_PACKAGE = ROOT / "model" if LOCAL else Path("/root/model")
MODULES_PACKAGE = ROOT / "modules" if LOCAL else Path("/root/modules")
FEATURES = ROOT / "results/sl_predict/features.npz" if LOCAL else Path("/root/data/features.npz")
FEATURES_V1 = ROOT / "results/sl_predict/features_v1.npz" if LOCAL else Path("/root/data/features_v1.npz")
FEATURES_CANCER = ROOT / "results/sl_predict/features_cancer.npz" if LOCAL else Path("/root/data/features_cancer.npz")
FEATURES_SPECTRAL = ROOT / "results/sl_predict/features_spectral.npz" if LOCAL else Path("/root/data/features_spectral.npz")
FEATURES_SPECTRAL_SAFE = ROOT / "results/sl_predict/features_spectral_safe.npz" if LOCAL else Path("/root/data/features_spectral_safe.npz")
FEATURES_SPECTRAL_SCGPT = ROOT / "results/sl_predict/features_spectral_scgpt.npz" if LOCAL else Path("/root/data/features_spectral_scgpt.npz")
SPLITS = ROOT / "results/sl_predict/splits.npz" if LOCAL else Path("/root/data/splits.npz")
SLKB = ROOT / "results/sl_predict/slkb_outcomes.npz" if LOCAL else Path("/root/data/slkb_outcomes.npz")
SLKB_EXTERNAL = ROOT / "results/sl_predict/slkb_outcomes_strict_external.npz" if LOCAL else Path("/root/data/slkb_outcomes_strict_external.npz")
SLKB_INTERVENTION = ROOT / "results/sl_predict/slkb_outcomes_intervention_external.npz" if LOCAL else Path("/root/data/slkb_outcomes_intervention_external.npz")
SLKB_CONTEXT = ROOT / "results/sl_predict/slkb_outcomes_intervention_context.npz" if LOCAL else Path("/root/data/slkb_outcomes_intervention_context.npz")
SLKB_DEPMAP = ROOT / "results/sl_predict/slkb_outcomes_intervention_depmap.npz" if LOCAL else Path("/root/data/slkb_outcomes_intervention_depmap.npz")
SLKB_DEPMAP_GENE = ROOT / "results/sl_predict/slkb_outcomes_intervention_depmap_gene.npz" if LOCAL else Path("/root/data/slkb_outcomes_intervention_depmap_gene.npz")
SLKB_DEPMAP_WORLD = ROOT / "results/sl_predict/slkb_outcomes_intervention_depmap_world.npz" if LOCAL else Path("/root/data/slkb_outcomes_intervention_depmap_world.npz")
DEPMAP_WORLD = ROOT / "results/sl_predict/depmap_world.npz" if LOCAL else Path("/root/data/depmap_world.npz")
DEPMAP_TOLERANCE = ROOT / "results/sl_predict/depmap_tolerance.npz" if LOCAL else Path("/root/data/depmap_tolerance.npz")
DEPENDENCY_LANDSCAPE = ROOT / "results/sl_predict/dependency_landscape.npz" if LOCAL else Path("/root/data/dependency_landscape.npz")
PERTURBSEQ_WORLD = ROOT / "results/sl_predict/perturbseq_world.npz" if LOCAL else Path("/root/data/perturbseq_world.npz")
PERTURBSEQ_WORLD_V2 = ROOT / "results/sl_predict/perturbseq_world_v2.npz" if LOCAL else Path("/root/data/perturbseq_world_v2.npz")
PERTURBSEQ_WORLD_V3 = ROOT / "results/sl_predict/perturbseq_world_v3.npz" if LOCAL else Path("/root/data/perturbseq_world_v3.npz")
PERTURBSEQ_NESTED96 = ROOT / "results/sl_predict/perturbseq_world_v3_nested96.npz" if LOCAL else Path("/root/data/perturbseq_world_v3_nested96.npz")
PERTURBSEQ_DIXIT = ROOT / "results/sl_predict/perturbseq_world_v4_dixit_context.npz" if LOCAL else Path("/root/data/perturbseq_world_v4_dixit_context.npz")
PERTURBSEQ_SOURCE_LANDMARK = ROOT / "results/sl_predict/perturbseq_source_landmark.npz" if LOCAL else Path("/root/data/perturbseq_source_landmark.npz")
BASAL_CONTEXT = ROOT / "results/sl_predict/basal_context.npz" if LOCAL else Path("/root/data/basal_context.npz")
HAP1_CONTEXT = ROOT / "results/sl_predict/hap1_context.npz" if LOCAL else Path("/root/data/hap1_context.npz")
PERTURBSEQ_FITNESS = ROOT / "results/sl_predict/perturbseq_world_fitness.npz" if LOCAL else Path("/root/data/perturbseq_world_fitness.npz")
SLKB_LABELS = ROOT / "results/sl_predict/slkb_labels.npz" if LOCAL else Path("/root/data/slkb_labels.npz")
BENCH = ROOT / "results/sl_predict/cv3_benchmarks.npz" if LOCAL else Path("/root/data/cv3_benchmarks.npz")
GRAPHS = ROOT / "results/sl_predict/relation_graphs.npz" if LOCAL else Path("/root/data/relation_graphs.npz")
EVALUATOR = ROOT / "src/benchmarks/sl_predict.py" if LOCAL else Path("/root/evaluate.py")
SLAMR_EVALUATOR = ROOT / "src/benchmarks/slamr.py" if LOCAL else Path("/root/slamr.py")
MUSL_EVALUATOR = ROOT / "src/benchmarks/musl.py" if LOCAL else Path("/root/musl.py")
TCGA_TREE_DECODER = ROOT / "src/training/tcga_tree_decoder.py" if LOCAL else Path("/root/tcga_tree_decoder.py")
HAP1_EVALUATOR = ROOT / "src/benchmarks/hap1.py" if LOCAL else Path("/root/hap1.py")
HAP1_PACK = ROOT / "results/sl_predict/hap1_score_pack.npz" if LOCAL else Path("/root/data/hap1_score_pack.npz")
HAP1_AUX = ROOT / "results/sl_predict/hap1_auxiliary.npz" if LOCAL else Path("/root/data/hap1_auxiliary.npz")
CODEPENDENCY = ROOT / "results/sl_predict/depmap_codependency.npz" if LOCAL else Path("/root/data/depmap_codependency.npz")
TCGA_RELATION = ROOT / "results/sl_predict/tcga_mutual_exclusivity.npz" if LOCAL else Path("/root/data/tcga_mutual_exclusivity.npz")
EXPRESSION_SILENCING = ROOT / "results/sl_predict/depmap_expression_silencing.npz" if LOCAL else Path("/root/data/depmap_expression_silencing.npz")
FULL_TRANSCRIPTOME_RESPONSE = ROOT / "results/sl_predict/full_transcriptome_codependency.npz" if LOCAL else Path("/root/data/full_transcriptome_codependency.npz")
META = ROOT / "data/feng2024/data/preprocessed_data/meta_table_9845.csv" if LOCAL else Path("/root/data/meta_table_9845.csv")
MUSL_META = ROOT / "data/models/MuSL/processed_data/meta_table_7684.csv" if LOCAL else Path("/root/data/musl/meta_table_7684.csv")
MUSL_FILES = [ROOT/f"data/models/MuSL/processed_data/data/CV3_bins_32/fold_data/{kind}_{part}_seed{seed}.pkl" if LOCAL else Path(f"/root/data/musl/{kind}_{part}_seed{seed}.pkl") for seed in (42,432) for kind in ("train","test") for part in ("pairs","labels")]
SLAMR_SPLITS = [ROOT / f"data/models/SLAMR/data_slb_filtered/{study}/{cell}_scenario3_fold5_seed88.pkl" if LOCAL else Path(f"/root/data/{cell}_scenario3_fold5_seed88.pkl") for study,cell in (("28319113","A549"),("30033366","JURKAT"),("30033366","K562"))]
SLAMR_TEXT = [ROOT / f"data/models/SLAMR/data_slb_filtered/LLM_emb/{study}_gpt-5.1_{cell}_desc_embedding.csv" if LOCAL else Path(f"/root/data/{cell}_text.csv") for study,cell in (("28319113","A549"),("30033366","JURKAT"),("30033366","K562"))]
app = modal.App("sl-predict")
volume = modal.Volume.from_name("sl-predict", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy==2.1.3", "scikit-learn==1.5.2", "lightgbm==4.5.0")
         .add_local_file(MODEL, "/root/world_model.py")
         .add_local_dir(MODEL_PACKAGE, "/root/model")
         .add_local_dir(MODULES_PACKAGE, "/root/modules")
         .add_local_file(FEATURES, "/root/data/features.npz")
         .add_local_file(FEATURES_V1, "/root/data/features_v1.npz")
         .add_local_file(FEATURES_CANCER, "/root/data/features_cancer.npz")
         .add_local_file(FEATURES_SPECTRAL, "/root/data/features_spectral.npz")
         .add_local_file(FEATURES_SPECTRAL_SAFE, "/root/data/features_spectral_safe.npz")
         .add_local_file(FEATURES_SPECTRAL_SCGPT, "/root/data/features_spectral_scgpt.npz")
         .add_local_file(SPLITS, "/root/data/splits.npz")
         .add_local_file(SLKB, "/root/data/slkb_outcomes.npz")
         .add_local_file(SLKB_EXTERNAL, "/root/data/slkb_outcomes_strict_external.npz")
         .add_local_file(SLKB_INTERVENTION, "/root/data/slkb_outcomes_intervention_external.npz")
         .add_local_file(SLKB_CONTEXT, "/root/data/slkb_outcomes_intervention_context.npz")
         .add_local_file(SLKB_DEPMAP, "/root/data/slkb_outcomes_intervention_depmap.npz")
         .add_local_file(SLKB_DEPMAP_GENE, "/root/data/slkb_outcomes_intervention_depmap_gene.npz")
         .add_local_file(SLKB_DEPMAP_WORLD, "/root/data/slkb_outcomes_intervention_depmap_world.npz")
         .add_local_file(DEPMAP_WORLD, "/root/data/depmap_world.npz")
         .add_local_file(DEPMAP_TOLERANCE, "/root/data/depmap_tolerance.npz")
         .add_local_file(DEPENDENCY_LANDSCAPE, "/root/data/dependency_landscape.npz")
         .add_local_file(PERTURBSEQ_WORLD, "/root/data/perturbseq_world.npz")
         .add_local_file(PERTURBSEQ_WORLD_V2, "/root/data/perturbseq_world_v2.npz")
         .add_local_file(PERTURBSEQ_WORLD_V3, "/root/data/perturbseq_world_v3.npz")
         .add_local_file(PERTURBSEQ_NESTED96, "/root/data/perturbseq_world_v3_nested96.npz")
         .add_local_file(PERTURBSEQ_DIXIT, "/root/data/perturbseq_world_v4_dixit_context.npz")
         .add_local_file(PERTURBSEQ_SOURCE_LANDMARK, "/root/data/perturbseq_source_landmark.npz")
         .add_local_file(BASAL_CONTEXT, "/root/data/basal_context.npz")
         .add_local_file(HAP1_CONTEXT, "/root/data/hap1_context.npz")
         .add_local_file(PERTURBSEQ_FITNESS, "/root/data/perturbseq_world_fitness.npz")
         .add_local_file(SLKB_LABELS, "/root/data/slkb_labels.npz")
         .add_local_file(GRAPHS, "/root/data/relation_graphs.npz"))
image=image.add_local_file(HAP1_EVALUATOR,"/root/hap1.py").add_local_file(HAP1_PACK,"/root/data/hap1_score_pack.npz").add_local_file(HAP1_AUX,"/root/data/hap1_auxiliary.npz")
eval_image = image.add_local_file(BENCH,"/root/data/cv3_benchmarks.npz").add_local_file(EVALUATOR,"/root/evaluate.py").add_local_file(SLAMR_EVALUATOR,"/root/slamr.py").add_local_file(MUSL_EVALUATOR,"/root/musl.py").add_local_file(META,"/root/data/meta_table_9845.csv").add_local_file(MUSL_META,"/root/data/musl/meta_table_7684.csv")
for split in SLAMR_SPLITS: eval_image=eval_image.add_local_file(split,f"/root/data/{split.name}")
for cell_text in SLAMR_TEXT: eval_image=eval_image.add_local_file(cell_text,f"/root/data/{cell_text.name.split('_gpt-5.1_')[-1].replace('_desc_embedding.csv','')}_text.csv")
for musl_file in MUSL_FILES: eval_image=eval_image.add_local_file(musl_file,f"/root/data/musl/{musl_file.name}")
public_eval_image=eval_image.add_local_file(CODEPENDENCY,"/root/data/depmap_codependency.npz").add_local_file(TCGA_RELATION,"/root/data/tcga_mutual_exclusivity.npz")
tcga_tree_image=image.add_local_file(TCGA_RELATION,"/root/data/tcga_mutual_exclusivity.npz").add_local_file(TCGA_TREE_DECODER,"/root/tcga_tree_decoder.py")
silencing_eval_image=public_eval_image.add_local_file(EXPRESSION_SILENCING,"/root/data/depmap_expression_silencing.npz")
response_eval_image=silencing_eval_image.add_local_file(FULL_TRANSCRIPTOME_RESPONSE,"/root/data/full_transcriptome_codependency.npz")


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/artifacts": volume})
def train(pretrain_epochs=12, rl_epochs=3, cv3_only=False, objective_only=False, d=384, latent=128, layers=6):
    import sys
    sys.path.insert(0, "/root")
    from modules.training.world import run
    name = f"native_d{d}_l{layers}_z{latent}_p{pretrain_epochs}_r{rl_epochs}"
    if cv3_only:
        raise ValueError("--cv3-only is retired; lock the checkpoint and invoke --evaluate-model separately.")
    cvs = ()
    rows = run("/root/data", f"/artifacts/{name}", pretrain_epochs, rl_epochs, cvs, d, latent, layers)
    volume.commit()
    return rows

@app.function(image=tcga_tree_image,gpu="L4",cpu=8,memory=32768,timeout=3600,volumes={"/artifacts":volume})
def train_tcga_tree(model_name="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"):
    import sys
    sys.path.insert(0,"/root"); from tcga_tree_decoder import fit
    result=fit(f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}"); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_relational(pretrain_epochs=12,rl_epochs=3,cancer=False):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_rel256{'_cancer' if cancer else ''}_guard_p{pretrain_epochs}_r{rl_epochs}"; feature_name="features_cancer.npz" if cancer else "features_v1.npz"; rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),384,128,6,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_spectral(pretrain_epochs=12,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_spectral_guard_p{pretrain_epochs}_r{rl_epochs}"; rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),384,128,6,"features_spectral.npz")
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_spectral_external(pretrain_epochs=12,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_spectral_external_guard_p{pretrain_epochs}_r{rl_epochs}"; rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),384,128,6,"features_spectral.npz","slkb_outcomes_strict_external.npz")
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_spectral_safe_external(pretrain_epochs=12,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_spectral_safe_external_guard_p{pretrain_epochs}_r{rl_epochs}"; rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),384,128,6,"features_spectral_safe.npz","slkb_outcomes_strict_external.npz")
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_spectral_safe_intervention(pretrain_epochs=12,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_spectral_safe_intervention_guard_p{pretrain_epochs}_r{rl_epochs}"; rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),384,128,6,"features_spectral_safe.npz","slkb_outcomes_intervention_external.npz")
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_safe_pretrain(pretrain_epochs=12):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run_pretrain
    name=f"native_spectral_safe_pretrain_p{pretrain_epochs}"; rows=run_pretrain("/root/data",f"/artifacts/{name}",pretrain_epochs,384,128,6,"features_spectral_safe.npz"); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_scgpt_pretrain(pretrain_epochs=12):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run_pretrain
    name=f"native_spectral_scgpt_pretrain_p{pretrain_epochs}"; rows=run_pretrain("/root/data",f"/artifacts/{name}",pretrain_epochs,384,128,6,"features_spectral_scgpt.npz"); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def train_safe_cold(mode="cold_joint",pretrain_epochs=12,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_outcomes
    name=f"native_spectral_safe_intervention_{mode}_p{pretrain_epochs}_r{rl_epochs}"; outcome="slkb_outcomes_intervention_depmap_gene.npz" if mode=="cold_depmap_gene" else ("slkb_outcomes_intervention_depmap.npz" if mode=="cold_depmap" else ("slkb_outcomes_intervention_context.npz" if mode=="cold_context" else "slkb_outcomes_intervention_external.npz")); rows=resume_outcomes("/root/data",f"/artifacts/native_spectral_safe_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",mode,6,rl_epochs,384,128,6,"features_spectral_safe.npz",outcome); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_safe_depmap_world(pretrain_epochs=12,dependency_epochs=3,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_depmap_world
    name=f"native_spectral_safe_intervention_depmap_world_p{pretrain_epochs}_d{dependency_epochs}_r{rl_epochs}"; rows=resume_depmap_world("/root/data",f"/artifacts/native_spectral_safe_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",dependency_epochs,6,rl_epochs); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_safe_perturbseq(pretrain_epochs=12,perturb_epochs=10,rl_epochs=3,residual=False,context_select=False,corpus="perturbseq_world.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_perturbseq
    name=f"native_spectral_safe_intervention_perturbseq{'_v2' if corpus.endswith('_v2.npz') else ''}{'_residual' if residual else ''}{'_context_select' if context_select else ''}_p{pretrain_epochs}_t{perturb_epochs}_r{rl_epochs}"; rows=resume_perturbseq("/root/data",f"/artifacts/native_spectral_safe_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",perturb_epochs,rl_epochs,6,residual_weight=float(residual),context_selection=context_select,perturb_name=corpus); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def train_basal_perturbseq(pretrain_epochs=12,dependency_epochs=3,perturb_epochs=10,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_basal_perturbseq
    name=f"native_spectral_safe_intervention_basal_perturbseq_v3_p{pretrain_epochs}_d{dependency_epochs}_t{perturb_epochs}_r{rl_epochs}"; rows=resume_basal_perturbseq("/root/data",f"/artifacts/native_spectral_safe_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",dependency_epochs,perturb_epochs,rl_epochs); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def train_scaled_pretrain():
    import sys
    sys.path.insert(0,"/root"); from world_model import run_pretrain
    name="native_spectral_safe_scaled_d768_z256_l8_p12"; result=run_pretrain("/root/data",f"/artifacts/{name}",12,768,256,8,"features_spectral_safe.npz"); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=14400,volumes={"/artifacts":volume})
def train_scaled_molecular():
    import sys
    sys.path.insert(0,"/root"); from world_model import resume_basal_perturbseq
    pre="native_spectral_safe_scaled_d768_z256_l8_p12"; name=f"{pre}_d3_t10_r3"; result=resume_basal_perturbseq("/root/data",f"/artifacts/{pre}/world_model.pt",f"/artifacts/{name}",3,10,3,0,768,256,8,evaluate_cv3=False,fit_outcome=False); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=14400,volumes={"/artifacts":volume})
def train_scaled_molecular_adjusted():
    import sys
    sys.path.insert(0,"/root"); from world_model import resume_basal_perturbseq
    pre="native_spectral_safe_scaled_d768_z256_l8_p12"; name=f"{pre}_adjusted_d3_t10_r3"; result=resume_basal_perturbseq("/root/data",f"/artifacts/{pre}/world_model.pt",f"/artifacts/{name}",3,10,3,0,768,256,8,evaluate_cv3=False,fit_outcome=False,perturb_learning_rate=0.00013411943852941884,perturb_rl_learning_rate=0.000008941295901961258); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def train_single_only_compact():
    import sys
    sys.path.insert(0,"/root"); from world_model import resume_basal_perturbseq
    pre="native_spectral_safe_pretrain_p12"; name=f"{pre}_single_only_d3_t10_r3"; result=resume_basal_perturbseq("/root/data",f"/artifacts/{pre}/world_model.pt",f"/artifacts/{name}",3,10,3,0,384,128,6,evaluate_cv3=False,fit_outcome=False,single_only=True); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=14400,volumes={"/artifacts":volume})
def train_single_only_scaled():
    import sys
    sys.path.insert(0,"/root"); from world_model import resume_basal_perturbseq
    pre="native_spectral_safe_scaled_d768_z256_l8_p12"; name=f"{pre}_single_only_d3_t10_r3"; result=resume_basal_perturbseq("/root/data",f"/artifacts/{pre}/world_model.pt",f"/artifacts/{name}",3,10,3,0,768,256,8,evaluate_cv3=False,fit_outcome=False,single_only=True); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_tolerance(model_name,epochs=5,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_tolerance_head
    result=fit_tolerance_head("/root/data",f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}",epochs,1000000,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_interaction(model_name,epochs=20,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_interaction_head
    result=fit_interaction_head("/root/data",f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}",epochs,100000,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_interaction_shrinkage(model_name,epochs=20,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_interaction_shrinkage_head
    result=fit_interaction_shrinkage_head("/root/data",f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}",epochs,100000,.15,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def compositional_ridge(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_compositional_ridge
    base="native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3"; residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_compositional_ridge("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/interaction_residual_head.pt",f"/artifacts/{residual}",d=d,latent=latent,layers=layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def source_landmark_endpoint(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_source_endpoint
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_source_endpoint("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",12,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dependency_landscape_endpoint(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dependency_landscape_endpoint
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dependency_landscape_endpoint("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",20,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dependency_core_endpoint(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dependency_landscape_endpoint
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dependency_landscape_endpoint("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",20,d,latent,layers,16,"dependency_core"); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dependency_action_adapter(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dependency_action_adapter
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dependency_action_adapter("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/dependency_landscape_endpoint.pt",f"/artifacts/{residual}",30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dependency_core_interaction(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dependency_core_interaction
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dependency_core_interaction("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/interaction_residual_head.pt",f"/artifacts/{residual}/dependency_core_endpoint.pt",f"/artifacts/{residual}",20,100000,.15,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def pair_transition(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_pair_transition_adapter
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_pair_transition_adapter("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dixit_pair_transition(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dixit_pair_adapter
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; out=f"/artifacts/{residual}/dixit_combinatorial"; result=fit_dixit_pair_adapter("/root/data",f"/artifacts/{residual}/world_model.pt",out,20,30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dixit_symmetric_latent(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dixit_symmetric_latent
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dixit_symmetric_latent("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/dixit_combinatorial/dixit_decoder.pt",f"/artifacts/{residual}/dixit_symmetric",30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def dixit_residual_latent(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_dixit_symmetric_latent
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_dixit_symmetric_latent("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/dixit_combinatorial/dixit_decoder.pt",f"/artifacts/{residual}/dixit_residual",30,d,latent,layers,True); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def transition_mixture(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_transition_mixture
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_transition_mixture("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def perturbseq_action_calibration(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_perturbseq_action_calibration
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_perturbseq_action_calibration("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/dependency_core_endpoint.pt",f"/artifacts/{residual}",20,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def single_trained_action_rotation(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_single_trained_action_rotation
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_single_trained_action_rotation("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/dependency_core_endpoint.pt",f"/artifacts/{residual}",30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def safe_symmetric_pair_fusion(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_safe_symmetric_pair_fusion
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_safe_symmetric_pair_fusion("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}",30,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def source_landmark_invariant_ridge(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_source_invariant_ridge
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; result=fit_source_invariant_ridge("/root/data",f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/interaction_residual_head.pt",f"/artifacts/{residual}/source_landmark_endpoint.pt",f"/artifacts/{residual}",d=d,latent=latent,layers=layers); volume.commit(); return result


@app.function(image=eval_image,gpu="L4",cpu=8,timeout=7200,volumes={"/artifacts":volume})
def source_landmark_emergent(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from musl import source_landmark_cv3
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; out=f"/artifacts/{residual}/source_landmark_emergent_musl.json"; result=source_landmark_cv3(f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/source_landmark_endpoint.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv","/root/data/basal_context.npz",out,d,latent,layers); volume.commit(); return result


@app.function(image=eval_image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def slamr_residual_interaction(d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from slamr import evaluate_residual_interaction
    residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; files=[f"/root/data/{cell}_scenario3_fold5_seed88.pkl" for cell in ("A549","JURKAT","K562")]; out=f"/artifacts/{residual}/slamr_residual_interaction.json"; result=evaluate_residual_interaction(f"/artifacts/{residual}/world_model.pt",f"/artifacts/{residual}/interaction_residual_head.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/basal_context.npz",files,out,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_interaction_depletion(model_name,epochs=20,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import fit_interaction_depletion_head
    result=fit_interaction_depletion_head("/root/data",f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}",epochs,100000,d,latent,layers); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_scgpt_perturbseq(pretrain_epochs=12,perturb_epochs=10,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_perturbseq
    name=f"native_spectral_scgpt_intervention_perturbseq_p{pretrain_epochs}_t{perturb_epochs}_r{rl_epochs}"; rows=resume_perturbseq("/root/data",f"/artifacts/native_spectral_scgpt_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",perturb_epochs,rl_epochs,6,feature_name="features_spectral_scgpt.npz"); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def train_safe_perturbseq_fitness(pretrain_epochs=12,perturb_epochs=10,rl_epochs=3):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_perturbseq
    name=f"native_spectral_safe_intervention_perturbseq_fitness_p{pretrain_epochs}_t{perturb_epochs}_r{rl_epochs}"; rows=resume_perturbseq("/root/data",f"/artifacts/native_spectral_safe_pretrain_p{pretrain_epochs}/world_model.pt",f"/artifacts/{name}",perturb_epochs,rl_epochs,6,384,128,6,"perturbseq_world_fitness.npz"); volume.commit(); return rows


@app.function(image=image,gpu="A100",timeout=7200,volumes={"/artifacts":volume})
def train_large(pretrain_epochs=12,rl_epochs=3,d=768,latent=256,layers=8):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import run
    name=f"native_d{d}_l{layers}_z{latent}_p{pretrain_epochs}_r{rl_epochs}"
    rows=run("/root/data",f"/artifacts/{name}",pretrain_epochs,rl_epochs,(),d,latent,layers)
    volume.commit(); return rows


@app.function(image=eval_image,gpu="A100",timeout=7200,volumes={"/artifacts":volume})
def benchmark(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from evaluate import evaluate
    out=f"/artifacts/{model_name}/cv3_benchmarks.json"
    rows=evaluate(f"/artifacts/{model_name}/world_model.pt","/root/data/features.npz","/root/data/cv3_benchmarks.npz",out,d,latent,layers)
    volume.commit(); return rows


@app.function(image=eval_image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def slamr(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from slamr import evaluate
    cells=("A549","JURKAT","K562"); files=[f"/root/data/{cell}_scenario3_fold5_seed88.pkl" for cell in cells]; texts=[f"/root/data/{cell}_text.csv" for cell in cells]; feature="features_spectral_safe.npz" if "safe" in model_name else "features_spectral.npz"; outcome="slkb_outcomes_intervention_depmap_world.npz" if "depmap_world" in model_name else ("slkb_outcomes_intervention_depmap_gene.npz" if "depmap_gene" in model_name else ("slkb_outcomes_intervention_depmap.npz" if "depmap" in model_name else ("slkb_outcomes_intervention_context.npz" if "cold_context" in model_name else ("slkb_outcomes_intervention_external.npz" if "intervention" in model_name else "slkb_outcomes_strict_external.npz")))); out=f"/artifacts/{model_name}/slamr_scenario3.json"; perturb=f"/artifacts/{model_name}/perturb_decoder.pt"; rows=evaluate(f"/artifacts/{model_name}/world_model.pt",f"/root/data/{feature}","/root/data/meta_table_9845.csv",files,out,f"/root/data/{outcome}",d,latent,layers,texts,perturb if Path(perturb).exists() else None); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def perturb_benchmark(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_perturb_evaluate
    rows=resume_perturb_evaluate("/root/data",f"/artifacts/{model_name}/world_model.pt",f"/artifacts/{model_name}/perturb_decoder.pt",f"/artifacts/{model_name}/metrics_rollout.json",d,latent,layers); volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def hap1_score_model(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from hap1 import score
    out=f"/artifacts/{model_name}/hap1_pair_scores.npz"; score(SimpleNamespace(pack="/root/data/hap1_score_pack.npz",model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",interaction_head=None,features="/root/data/features_spectral_safe.npz",output=out,context_pack=None,context_model="ACH-002475",batch=8192,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def hap1_score_basal(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from hap1 import score
    out=f"/artifacts/{model_name}/hap1_basal_pair_scores.npz"; score(SimpleNamespace(pack="/root/data/hap1_score_pack.npz",model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",interaction_head=None,features="/root/data/features_spectral_safe.npz",output=out,context_pack="/root/data/hap1_context.npz",context_model="ACH-002475",batch=8192,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def hap1_score_interaction(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from hap1 import score
    out=f"/artifacts/{model_name}/hap1_interaction_pair_scores.npz"; score(SimpleNamespace(pack="/root/data/hap1_score_pack.npz",model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",interaction_head=f"/artifacts/{model_name}/interaction_head.pt",features="/root/data/features_spectral_safe.npz",output=out,context_pack="/root/data/hap1_context.npz",context_model="ACH-002475",batch=8192,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=image,gpu="L4",timeout=7200,volumes={"/artifacts":volume})
def hap1_score_interaction_shrinkage(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from hap1 import score
    out=f"/artifacts/{model_name}/hap1_interaction_shrinkage_pair_scores.npz"; score(SimpleNamespace(pack="/root/data/hap1_score_pack.npz",model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",interaction_head=f"/artifacts/{model_name}/interaction_shrinkage_head.pt",features="/root/data/features_spectral_safe.npz",output=out,context_pack="/root/data/hap1_context.npz",context_model="ACH-002475",batch=8192,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=image,gpu="L4",timeout=5400,volumes={"/artifacts":volume})
def hap1_score_residual_endpoint(d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from hap1 import score
    base="native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3"; residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; out=f"/artifacts/{residual}/hap1_residual_pair_scores.npz"; score(SimpleNamespace(pack="/root/data/hap1_score_pack.npz",model=f"/artifacts/{base}/world_model.pt",decoder=f"/artifacts/{base}/perturb_decoder.pt",interaction_head=f"/artifacts/{base}/interaction_shrinkage_head.pt",residual_model=f"/artifacts/{residual}/world_model.pt",residual_interaction_head=f"/artifacts/{residual}/interaction_residual_head.pt",features="/root/data/features_spectral_safe.npz",output=out,context_pack="/root/data/hap1_context.npz",context_model="ACH-002475",batch=8192,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",cpu=8,timeout=7200,volumes={"/artifacts":volume})
def musl_calibrated(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers); volume.commit(); return result


@app.function(image=public_eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_calibrated_public(model_name: str,d: int=768,latent: int=256,layers: int=8):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated_public.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,codependency_path="/root/data/depmap_codependency.npz",tcga_path="/root/data/tcga_mutual_exclusivity.npz"); volume.commit(); return result


@app.function(image=silencing_eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_calibrated_public_silencing(model_name: str,d: int=768,latent: int=256,layers: int=8):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated_public_silencing.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,codependency_path="/root/data/depmap_codependency.npz",tcga_path="/root/data/tcga_mutual_exclusivity.npz",silencing_path="/root/data/depmap_expression_silencing.npz"); volume.commit(); return result


@app.function(image=silencing_eval_image,gpu="L4",cpu=8,memory=32768,timeout=14400,volumes={"/artifacts":volume})
def musl_calibrated_public_silencing_stacked(model_name: str,d: int=768,latent: int=256,layers: int=8):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated_public_silencing_stacked.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,codependency_path="/root/data/depmap_codependency.npz",tcga_path="/root/data/tcga_mutual_exclusivity.npz",silencing_path="/root/data/depmap_expression_silencing.npz",stacked=True); volume.commit(); return result


@app.function(image=response_eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_calibrated_full_transcriptome(model_name: str,d: int=768,latent: int=256,layers: int=8):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated_full_transcriptome.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,codependency_path="/root/data/depmap_codependency.npz",tcga_path="/root/data/tcga_mutual_exclusivity.npz",silencing_path="/root/data/depmap_expression_silencing.npz",response_path="/root/data/full_transcriptome_codependency.npz"); volume.commit(); return result


@app.function(image=eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_calibrated_tolerance(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_calibrated_tolerance.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,tolerance_path=f"/artifacts/{model_name}/tolerance_head.pt",context_pack="/root/data/basal_context.npz"); volume.commit(); return result


@app.function(image=eval_image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def musl_basal(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from musl import run
    out=f"/artifacts/{model_name}/musl_cv3_basal_risk.json"; run(SimpleNamespace(model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",features="/root/data/features_spectral_safe.npz",meta="/root/data/meta_table_9845.csv",musl_meta="/root/data/musl/meta_table_7684.csv",output=out,seeds=(42,432),context_pack="/root/data/basal_context.npz",tolerance_head=None,interaction_head=None,contexts=32,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",timeout=5400,volumes={"/artifacts":volume})
def musl_tolerance(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from musl import run
    out=f"/artifacts/{model_name}/musl_cv3_tolerance.json"; run(SimpleNamespace(model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",features="/root/data/features_spectral_safe.npz",meta="/root/data/meta_table_9845.csv",musl_meta="/root/data/musl/meta_table_7684.csv",output=out,seeds=(42,432),context_pack="/root/data/basal_context.npz",tolerance_head=f"/artifacts/{model_name}/tolerance_head.pt",interaction_head=None,contexts=32,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",timeout=5400,volumes={"/artifacts":volume})
def musl_interaction(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from musl import run
    out=f"/artifacts/{model_name}/musl_cv3_interaction.json"; run(SimpleNamespace(model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",features="/root/data/features_spectral_safe.npz",meta="/root/data/meta_table_9845.csv",musl_meta="/root/data/musl/meta_table_7684.csv",output=out,seeds=(42,432),context_pack="/root/data/basal_context.npz",tolerance_head=None,interaction_head=f"/artifacts/{model_name}/interaction_head.pt",contexts=32,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",timeout=5400,volumes={"/artifacts":volume})
def musl_interaction_shrinkage(model_name,d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from musl import run
    out=f"/artifacts/{model_name}/musl_cv3_interaction_shrinkage.json"; run(SimpleNamespace(model=f"/artifacts/{model_name}/world_model.pt",decoder=f"/artifacts/{model_name}/perturb_decoder.pt",features="/root/data/features_spectral_safe.npz",meta="/root/data/meta_table_9845.csv",musl_meta="/root/data/musl/meta_table_7684.csv",output=out,seeds=(42,432),context_pack="/root/data/basal_context.npz",tolerance_head=None,interaction_head=f"/artifacts/{model_name}/interaction_shrinkage_head.pt",contexts=32,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",timeout=5400,volumes={"/artifacts":volume})
def musl_residual_endpoint(d=384,latent=128,layers=6):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0,"/root"); from musl import run
    base="native_spectral_safe_intervention_basal_perturbseq_v3_p12_d3_t10_r3"; residual="native_spectral_safe_intervention_basal_perturbseq_residual64_p12_d3_t10_r3"; out=f"/artifacts/{residual}/musl_cv3_residual_endpoint.json"; run(SimpleNamespace(model=f"/artifacts/{base}/world_model.pt",decoder=f"/artifacts/{base}/perturb_decoder.pt",features="/root/data/features_spectral_safe.npz",meta="/root/data/meta_table_9845.csv",musl_meta="/root/data/musl/meta_table_7684.csv",output=out,seeds=(42,432),context_pack="/root/data/basal_context.npz",tolerance_head=None,interaction_head=f"/artifacts/{base}/interaction_shrinkage_head.pt",residual_model=f"/artifacts/{residual}/world_model.pt",residual_interaction_head=f"/artifacts/{residual}/interaction_residual_head.pt",contexts=32,d=d,latent=latent,layers=layers)); volume.commit(); return out


@app.function(image=eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_hap1(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_hap1_auxiliary.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,aux_path="/root/data/hap1_auxiliary.npz"); volume.commit(); return result


@app.function(image=eval_image,gpu="L4",cpu=8,memory=16384,timeout=7200,volumes={"/artifacts":volume})
def musl_hap1_sequential(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from musl import calibrated
    out=f"/artifacts/{model_name}/musl_cv3_hap1_sequential.json"; result=calibrated(f"/artifacts/{model_name}/world_model.pt","/root/data/features_spectral_safe.npz","/root/data/meta_table_9845.csv","/root/data/musl/meta_table_7684.csv",out,d,latent,layers,aux_path="/root/data/hap1_auxiliary.npz",aux_sequential=True); volume.commit(); return result


@app.function(image=image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def calibrate(model_name,d=384,latent=128,layers=6):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume
    out=f"/artifacts/{model_name}/rich_metrics.json"; rows=resume("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def metric(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_metric
    out=f"/artifacts/{model_name}/metric_metrics.json"; rows=resume_metric("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def pair(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_pair
    out=f"/artifacts/{model_name}/pair_metrics.json"; rows=resume_pair("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=3600,volumes={"/artifacts":volume})
def emergent(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_emergent
    out=f"/artifacts/{model_name}/emergent_metrics.json"; rows=resume_emergent("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",timeout=1800,volumes={"/artifacts":volume})
def transfer(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_transfer
    out=f"/artifacts/{model_name}/transfer_metrics.json"; rows=resume_transfer("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",cpu=8,timeout=3600,volumes={"/artifacts":volume})
def tabular(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_tabular
    out=f"/artifacts/{model_name}/tabular_metrics.json"; rows=resume_tabular("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.function(image=image,gpu="L4",cpu=8,timeout=3600,volumes={"/artifacts":volume})
def graph(model_name,d=384,latent=128,layers=6,feature_name="features.npz"):
    import sys
    sys.path.insert(0,"/root"); from modules.training.world import resume_graph
    out=f"/artifacts/{model_name}/graph_metrics.json"; rows=resume_graph("/root/data",f"/artifacts/{model_name}/world_model.pt",out,d,latent,layers,feature_name)
    volume.commit(); return rows


@app.local_entrypoint()
def main(pretrain_epochs: int = 12, rl_epochs: int = 3, cv3_only: bool = False,
         objective_only: bool = False, evaluate_model: str = "", large: bool = False,
         calibrate_model: str = "", musl_calibrated_tolerance_model: str = "", metric_model: str = "", pair_model: str = "", emergent_model: str = "", transfer_model: str = "", tabular_model: str = "", graph_model: str = "", slamr_model: str = "", perturb_model: str = "", musl_model: str = "", musl_basal_model: str = "", musl_tolerance_model: str = "", musl_interaction_model: str = "", musl_interaction_shrinkage_model: str = "", musl_hap1_model: str = "", musl_hap1_sequential_model: str = "", hap1_model: str = "", hap1_basal_model: str = "", hap1_interaction_model: str = "", hap1_interaction_shrinkage_model: str = "", tolerance_model: str = "", interaction_model: str = "", interaction_shrinkage_model: str = "", interaction_depletion_model: str = "", relational: bool = False, cancer: bool = False, spectral: bool = False, spectral_external: bool = False, spectral_safe_external: bool = False, spectral_safe_intervention: bool = False, safe_pretrain: bool = False, scgpt_pretrain: bool = False, safe_cold: str = "", depmap_world: bool = False, dependency_epochs: int = 3, perturbseq: bool = False, perturbseq_v2: bool = False, basal_perturbseq: bool = False, perturbseq_residual: bool = False, perturbseq_context_select: bool = False, perturbseq_fitness: bool = False, scgpt_perturbseq: bool = False, perturb_epochs: int = 10, d: int = 384, latent: int = 128, layers: int = 6):
    model_name=evaluate_model or calibrate_model or musl_calibrated_tolerance_model or metric_model or pair_model or emergent_model or transfer_model or tabular_model or graph_model or slamr_model or perturb_model or musl_model or musl_basal_model or musl_tolerance_model or musl_interaction_model or musl_interaction_shrinkage_model or musl_hap1_model or musl_hap1_sequential_model or hap1_model or hap1_basal_model or hap1_interaction_model or hap1_interaction_shrinkage_model or tolerance_model or interaction_model or interaction_shrinkage_model or interaction_depletion_model; feature_name="features_spectral_safe.npz" if "safe" in model_name else ("features_spectral.npz" if "spectral" in model_name else ("features_cancer.npz" if "cancer" in model_name else ("features_v1.npz" if "rel256" in model_name else "features.npz")))
    if evaluate_model: print(benchmark.remote(evaluate_model,d,latent,layers))
    elif calibrate_model: print(calibrate.remote(calibrate_model,d,latent,layers))
    elif musl_calibrated_tolerance_model: print(musl_calibrated_tolerance.remote(musl_calibrated_tolerance_model,d,latent,layers))
    elif metric_model: print(metric.remote(metric_model,d,latent,layers,feature_name))
    elif pair_model: print(pair.remote(pair_model,d,latent,layers,feature_name))
    elif emergent_model: print(emergent.remote(emergent_model,d,latent,layers,feature_name))
    elif transfer_model: print(transfer.remote(transfer_model,d,latent,layers,feature_name))
    elif tabular_model: print(tabular.remote(tabular_model,d,latent,layers,feature_name))
    elif graph_model: print(graph.remote(graph_model,d,latent,layers,feature_name))
    elif slamr_model: print(slamr.remote(slamr_model,d,latent,layers))
    elif perturb_model: print(perturb_benchmark.remote(perturb_model,d,latent,layers))
    elif musl_model: print(musl_calibrated.remote(musl_model,d,latent,layers))
    elif musl_basal_model: print(musl_basal.remote(musl_basal_model,d,latent,layers))
    elif musl_tolerance_model: print(musl_tolerance.remote(musl_tolerance_model,d,latent,layers))
    elif musl_interaction_model: print(musl_interaction.remote(musl_interaction_model,d,latent,layers))
    elif musl_interaction_shrinkage_model: print(musl_interaction_shrinkage.remote(musl_interaction_shrinkage_model,d,latent,layers))
    elif musl_hap1_model: print(musl_hap1.remote(musl_hap1_model,d,latent,layers))
    elif musl_hap1_sequential_model: print(musl_hap1_sequential.remote(musl_hap1_sequential_model,d,latent,layers))
    elif hap1_model: print(hap1_score_model.remote(hap1_model,d,latent,layers))
    elif hap1_basal_model: print(hap1_score_basal.remote(hap1_basal_model,d,latent,layers))
    elif hap1_interaction_model: print(hap1_score_interaction.remote(hap1_interaction_model,d,latent,layers))
    elif hap1_interaction_shrinkage_model: print(hap1_score_interaction_shrinkage.remote(hap1_interaction_shrinkage_model,d,latent,layers))
    elif tolerance_model: print(train_tolerance.remote(tolerance_model,5,d,latent,layers))
    elif interaction_model: print(train_interaction.remote(interaction_model,20,d,latent,layers))
    elif interaction_shrinkage_model: print(train_interaction_shrinkage.remote(interaction_shrinkage_model,20,d,latent,layers))
    elif interaction_depletion_model: print(train_interaction_depletion.remote(interaction_depletion_model,20,d,latent,layers))
    elif spectral_external: print(train_spectral_external.remote(pretrain_epochs,rl_epochs))
    elif spectral_safe_external: print(train_spectral_safe_external.remote(pretrain_epochs,rl_epochs))
    elif spectral_safe_intervention: print(train_spectral_safe_intervention.remote(pretrain_epochs,rl_epochs))
    elif safe_pretrain: print(train_safe_pretrain.remote(pretrain_epochs))
    elif scgpt_pretrain: print(train_scgpt_pretrain.remote(pretrain_epochs))
    elif safe_cold: print(train_safe_cold.remote(safe_cold,pretrain_epochs,rl_epochs))
    elif depmap_world: print(train_safe_depmap_world.remote(pretrain_epochs,dependency_epochs,rl_epochs))
    elif basal_perturbseq: print(train_basal_perturbseq.remote(pretrain_epochs,dependency_epochs,perturb_epochs,rl_epochs))
    elif perturbseq or perturbseq_v2 or perturbseq_residual or perturbseq_context_select: print(train_safe_perturbseq.remote(pretrain_epochs,perturb_epochs,rl_epochs,perturbseq_residual,perturbseq_context_select,"perturbseq_world_v2.npz" if perturbseq_v2 else "perturbseq_world.npz"))
    elif perturbseq_fitness: print(train_safe_perturbseq_fitness.remote(pretrain_epochs,perturb_epochs,rl_epochs))
    elif scgpt_perturbseq: print(train_scgpt_perturbseq.remote(pretrain_epochs,perturb_epochs,rl_epochs))
    elif spectral: print(train_spectral.remote(pretrain_epochs,rl_epochs))
    elif relational: print(train_relational.remote(pretrain_epochs,rl_epochs,cancer))
    elif large: print(train_large.remote(pretrain_epochs,rl_epochs,768,256,8))
    else: print(train.remote(pretrain_epochs, rl_epochs, cv3_only, objective_only, d, latent, layers))
