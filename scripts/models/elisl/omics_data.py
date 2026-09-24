"""Single-gene omics matrices for ELISL's context-specific features (all cached as parquet in _slb/cache/).

Cell lines (DepMap 24Q4; ELISL used DepMap 18Q3 CRISPR `gene_dependency.csv` + CCLE 2019 via cBioPortal):
  dep        CRISPRGeneDependency (probability of dependency, the 24Q4 analogue of 18Q3 gene_dependency.csv)
  cl_mut     non-silent somatic mutations (OmicsSomaticMutations, protein-altering / splice consequences)
  cl_exprz   expression z-score per gene across all lines (log2 TPM+1; ELISL: cBioPortal median_all_sample_Zscores)
Tissue (TCGA PanCanAtlas via UCSC Xena; ELISL used per-study cBioPortal/Firehose files for the same TCGA samples):
  tcga_expr  EB++ batch-corrected RSEM, log2(x+1), tumour AND normal samples (normal = sample type 10-19)
  tcga_cna   GISTIC2 thresholded (-2..2), tcga_mut MC3 non-silent gene-level (0/1)
  tcga_clin  TCGA-CDR: type, age, gender, race, OS, OS.time
GTEx v8 gene TPM (healthy co-expression).
"""
import numpy as np
import pandas as pd

import common as C

NONSILENT = ("missense", "frameshift", "stop_gained", "stop_lost", "start_lost", "splice_acceptor", "splice_donor",
             "inframe", "protein_altering")


def _depmap_csv(fname):
    df = pd.read_csv(C.RAW / "depmap" / fname, index_col=0)
    df.columns = C.resolve_cols(df.columns)
    return C.dedupe_cols(df).astype("float32")


def dep():
    return C.cached("dep_prob", lambda: _depmap_csv("CRISPRGeneDependency_24Q4.csv"))


def cl_mut():
    def b():
        d = pd.read_csv(C.RAW / "depmap/OmicsSomaticMutations_24Q4.csv",
                        usecols=["ModelID", "HugoSymbol", "EntrezGeneID", "VariantInfo"])
        d = d[d.VariantInfo.fillna("").str.contains("|".join(NONSILENT))]
        res = C.resolver()
        sym = [res.get(str(int(e))) if e == e else None for e in d.EntrezGeneID]
        d["g"] = [s or res.get(h) for s, h in zip(sym, d.HugoSymbol)]
        d = d.dropna(subset=["g"])[["ModelID", "g"]].drop_duplicates()
        m = pd.crosstab(d.ModelID, d.g).clip(upper=1).astype("float32")
        return m
    return C.cached("cl_mut_nonsilent", b)


def cl_exprz():
    def b():
        df = pd.read_csv(C.RAW / "depmap/OmicsExpressionProteinCodingGenesTPMLogp1_24Q4.csv", index_col=0)
        df = df.select_dtypes("number")
        df.columns = C.resolve_cols(df.columns)
        df = C.dedupe_cols(df).astype("float32")
        return ((df - df.mean()) / df.std()).astype("float32")
    return C.cached("cl_exprz", b)


def _xena(fname):
    df = pd.read_csv(C.RAW / "tcga_pancan" / fname, sep="\t", index_col=0)
    df.index = C.resolve_cols(df.index)
    df = df[[i is not None for i in df.index]]
    df = df[~df.index.duplicated()]
    return df.T.astype("float32")


def tcga_expr():
    return C.cached("tcga_expr_log2_all", lambda: _xena("EBpp_geneExp.xena.gz"))


def tcga_cna():
    return C.cached("tcga_cna", lambda: _xena("gistic2_thresholded.by_genes.gz"))


def tcga_mut():
    return C.cached("tcga_mut", lambda: _xena("mc3_nonsilentGene.xena.gz"))


def tcga_clin():
    def b():
        s = pd.read_csv(C.RAW / "tcga_pancan/survival_S1_20171025.tsv", sep="\t")
        s = s.rename(columns={"cancer type abbreviation": "type", "age_at_initial_pathologic_diagnosis": "age",
                              "_PATIENT": "patient"})
        return s[["sample", "patient", "type", "age", "gender", "race", "OS", "OS.time"]].set_index("sample")
    return C.cached("tcga_clin", b)


def patient_type() -> pd.Series:
    c = tcga_clin()
    return c.drop_duplicates("patient").set_index("patient")["type"]


def is_tumour(samples) -> np.ndarray:
    return np.array([len(s) >= 15 and s[13:15].isdigit() and int(s[13:15]) < 10 for s in samples])


def is_normal(samples) -> np.ndarray:
    return np.array([len(s) >= 15 and s[13:15].isdigit() and 10 <= int(s[13:15]) < 20 for s in samples])


GTEX_TISSUE = {  # ELISL's cancer2tissue for its 8 types + analogous choices for the extra SLB types
    "BRCA": ["Breast - Mammary Tissue"], "CESC": ["Cervix - Ectocervix", "Cervix - Endocervix"],
    "COAD": ["Colon - Transverse"], "READ": ["Colon - Transverse"], "KIRC": ["Kidney - Cortex"],
    "LAML": ["Whole Blood"], "LUAD": ["Lung"], "LUSC": ["Lung"], "OV": ["Ovary"],
    "SKCM": ["Skin - Not Sun Exposed (Suprapubic)", "Skin - Sun Exposed (Lower leg)"], "PAAD": ["Pancreas"],
    "HNSC": ["Minor Salivary Gland"], "STAD": ["Stomach"], "ESCA": ["Esophagus - Mucosa"], "LIHC": ["Liver"],
    "PRAD": ["Prostate"], "THCA": ["Thyroid"], "UCEC": ["Uterus"], "BLCA": ["Bladder"],
    "GBM": ["Brain - Cortex"], "LGG": ["Brain - Cortex"],
}


def gtex():
    """GTEx v8 TPM, samples x protein-coding HGNC genes (float32)."""
    def b():
        h = pd.read_csv(C.RAW / "ids/hgnc_complete_set.txt", sep="\t", low_memory=False, usecols=["symbol", "locus_group"])
        pc = set(h.symbol[h.locus_group == "protein-coding gene"])
        res = C.resolver()
        parts = []
        for ch in pd.read_csv(C.RAW / "gtex_v8/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz", sep="\t",
                              skiprows=2, chunksize=2000):
            ens = ch.Name.str.split(".").str[0]
            sym = [res.get(e) or res.get(d) for e, d in zip(ens, ch.Description)]
            ch.index = sym
            ch = ch[[s is not None and s in pc for s in sym]].drop(columns=["Name", "Description"])
            parts.append(ch.astype("float32"))
        df = pd.concat(parts)
        # duplicated symbols: keep the most variable row (as ELISL does for duplicated columns)
        df["_v"] = df.var(axis=1)
        df = df.sort_values("_v", ascending=False)
        df = df[~df.index.duplicated()].drop(columns="_v")
        return df.T
    return C.cached("gtex_tpm_pc", b)


def gtex_samples(cancer) -> list:
    a = pd.read_csv(C.RAW / "gtex_v8/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep="\t",
                    usecols=["SAMPID", "SMTSD"])
    if cancer in GTEX_TISSUE:
        return a.SAMPID[a.SMTSD.isin(GTEX_TISSUE[cancer])].tolist()
    return a.SAMPID.tolist()  # PANCAN / unmapped: all GTEx samples
