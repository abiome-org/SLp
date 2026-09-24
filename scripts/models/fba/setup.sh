#!/usr/bin/env bash
# One-time setup for the FBA battery model: venv (COBRApy + GLPK/HiGHS), GEM downloads, annotation files.
# Everything goes to external/models/fba (gitignored). Checksums of the versions used: external/models/fba/SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=external/models/fba
mkdir -p $E/gems $E/annot $E/bin
[ -x $E/.venv/bin/python ] || { uv venv -q --python 3.11 $E/.venv; VIRTUAL_ENV=$E/.venv uv pip install -q cobra==0.32.1 highspy pandas pyarrow; }
cd $E/gems
[ -d yeast-GEM ] || git clone -q --depth 1 https://github.com/SysBioChalmers/yeast-GEM.git      # Yeast9 lineage
[ -d Human-GEM ] || git clone -q --depth 1 https://github.com/SysBioChalmers/Human-GEM.git
[ -f pombeModellingSupplements.zip ] || { curl -sL -o pombeModellingSupplements.zip \
    https://zenodo.org/api/records/6513463/files/pombeModellingSupplements.zip/content; unzip -q -o pombeModellingSupplements.zip 'models/*' -d pombe; }
[ -f iDS372_Data_Sheet_2.zip ] || { curl -sL -A Mozilla/5.0 -o iDS372_Data_Sheet_2.zip \
    "https://www.frontiersin.org/api/v4/articles/457856/file/Data_Sheet_2.zip/457856_supplementary-materials_datasheets_2_zip/1"; unzip -o iDS372_Data_Sheet_2.zip -d iDS372; }
for m in iML1515 iYO844; do [ -f $m.xml.gz ] || curl -sL -o $m.xml.gz http://bigg.ucsd.edu/static/models/$m.xml.gz; done
cd ../annot
E2=https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
[ -f D39V_CP027540.gb ] || curl -sL -o D39V_CP027540.gb "$E2?db=nuccore&id=CP027540.1&rettype=gbwithparts&retmode=text"
[ -f R6_AE007317.gb ] || curl -sL -o R6_AE007317.gb "$E2?db=nuccore&id=AE007317.1&rettype=gbwithparts&retmode=text"
cd ../../../..
# R6 (iDS372 genes) -> D39V reciprocal best hits with DIAMOND
if [ ! -f $E/annot/R6_vs_D39V.tsv ]; then
  [ -x $E/bin/diamond ] || curl -sL https://github.com/bbuchfink/diamond/releases/download/v2.1.10/diamond-linux64.tar.gz | tar xz -C $E/bin
  $E/.venv/bin/python scripts/models/fba/genbank_faa.py
  (cd $E/annot && ../bin/diamond makedb --in D39V.faa -d D39V --quiet && ../bin/diamond makedb --in R6.faa -d R6 --quiet &&
   ../bin/diamond blastp -q R6.faa -d D39V -o R6_vs_D39V.tsv -k 1 -p 8 --quiet --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore &&
   ../bin/diamond blastp -q D39V.faa -d R6 -o D39V_vs_R6.tsv -k 1 -p 8 --quiet --outfmt 6 qseqid sseqid pident length qlen slen evalue bitscore)
fi
# canonical spne IDs (slpbench.ids_extra via bacteria_extra.spne_resolver) exported for the cobra venv
uv run python -c "
from slpbench.sources.bacteria_extra import spne_resolver
r=spne_resolver(); open('$E/annot/spne_resolver.tsv','w').write('alias\tcanonical\n'+''.join(f'{k}\t{v}\n' for k,v in r.items()))"
