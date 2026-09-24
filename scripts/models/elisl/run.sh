#!/usr/bin/env bash
# ELISL (Tepeli, Seale, Goncalves, Bioinformatics 2024) retrained on SLB train labels only.
#   elisl__slbtrain : ELISL feature pipeline rebuilt for SLB human pairs (SeqVec, node2vec PPI, DepMap cell-line
#                     dependency x mutation/expression, TCGA/GTEx tissue features) + original ELRRF class.
# Human-only by construction (cell-line/TCGA/GTEx inputs); other species are left missing (median-filled by eval).
# env: SLB_BENCH (default data/bench/slb1.2), SLB_SPLIT (default dev), ELISL_CPUS (default 8, house limit),
#      ELISL_GPU=1 to run SeqVec on the GPU (claim it on the board first; CPU SeqVec is ~100 residues/s).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export SLB_BENCH=${SLB_BENCH:-data/bench/slb1.2}
SPLIT=${SLB_SPLIT:-${1:-dev}}
export ELISL_CPUS=${ELISL_CPUS:-8}
export OMP_NUM_THREADS=$ELISL_CPUS OPENBLAS_NUM_THREADS=$ELISL_CPUS MKL_NUM_THREADS=$ELISL_CPUS NUMEXPR_NUM_THREADS=$ELISL_CPUS
M=external/models/elisl
D=scripts/models/elisl
PY=$ROOT/$M/.venv/bin/python
NAME=elisl__slbtrain

# --- environments
if [ ! -x "$PY" ]; then
  uv venv --python 3.8 $M/.venv
  VIRTUAL_ENV=$M/.venv uv pip install -r $D/requirements.lock.txt
fi
docker image inspect slb/elisl-seqvec >/dev/null 2>&1 || \
  docker build -t slb/elisl-seqvec -f $M/slb_docker/Dockerfile.seqvec $M/slb_docker

# --- raw single-gene inputs (fetched once; URL + sha256 in data/raw/<key>/SOURCES.tsv)
F="uv run python scripts/models/_common/fetch.py"
$F string_v11 https://stringdb-static.org/download/protein.links.full.v11.0/9606.protein.links.full.v11.0.txt.gz >/dev/null
$F string_v11 https://stringdb-static.org/download/protein.info.v11.0/9606.protein.info.v11.0.txt.gz >/dev/null
$F seqvec http://data.bioembeddings.com/public/embeddings/embedding_models/seqvec/options.json >/dev/null
$F seqvec http://data.bioembeddings.com/public/embeddings/embedding_models/seqvec/weights.hdf5 >/dev/null
$F uniprot_human "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28reviewed%3Atrue%29+AND+%28organism_id%3A9606%29" uniprot_reviewed_9606.fasta >/dev/null
$F uniprot_human "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession%2Cid%2Cgene_primary%2Cgene_synonym%2Cxref_hgnc&query=%28reviewed%3Atrue%29+AND+%28organism_id%3A9606%29" uniprot_reviewed_9606_genes.tsv >/dev/null
$F gtex_v8 https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt >/dev/null
$F gtex_v8 https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz >/dev/null
$F depmap https://ndownloader.figshare.com/files/51064631 CRISPRGeneDependency_24Q4.csv >/dev/null
$F depmap https://ndownloader.figshare.com/files/51065732 OmicsSomaticMutations_24Q4.csv >/dev/null
# also required (shared, fetched by other adapters): data/raw/depmap/{Model,OmicsExpressionProteinCodingGenesTPMLogp1}_24Q4.csv,
# data/raw/tcga_pancan/*, data/raw/ids/hgnc_complete_set.txt, data/interim/bundle/human/ppi.parquet (BioGRID physical)

# --- features (per-gene / per-(pair, cancer) caches in $M/_slb/cache make re-runs incremental)
cd $D
$PY seq_embed.py train "$SPLIT"
$PY ppi_embed.py
$PY features.py train
[ "$SPLIT" = train ] || $PY features.py "$SPLIT"
mkdir -p "$ROOT/$M/_slb/pred/$(basename $SLB_BENCH)"
PRED="$ROOT/$M/_slb/pred/$(basename $SLB_BENCH)/${NAME}_${SPLIT}.parquet"
$PY elrrf.py "$SPLIT" "$PRED"
cd "$ROOT"
# shared writer: fills species ELISL cannot score with a constant, writes .coverage.json, evaluates dev
# (results/models/ for slb1.2, results/models/<bench>/ otherwise)
uv run python $D/finalize.py "$PRED" $NAME "$SPLIT"
