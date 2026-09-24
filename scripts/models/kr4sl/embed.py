"""CODER (GanjinZero/UMLSBert_ENG) CLS embeddings of every KR4SL entity name, exactly as KR4SL's
data/extract_pretrain_emb.py (max_len 512, padded, no attention mask, L2-normalised), on GPU if available."""
import os, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("SLB_BENCH", ROOT / "data/bench/slb1.2"))
WORK = Path(os.environ.get("SLB_WORK", ROOT / "external/models/_slb_work" / BENCH.name))
D = WORK / "kr4sl"
dev = "cuda" if torch.cuda.is_available() and os.environ.get("SLB_GPU") == "1" else "cpu"
torch.set_num_threads(int(os.environ.get("SLB_THREADS", "8")))
m = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG").to(dev).eval()
tok = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
ents = [l.strip() for l in open(D / "all_entities.txt")]
out = []
with torch.no_grad():
    for s in range(0, len(ents), 64):
        ids = [tok.encode_plus(p, max_length=512, add_special_tokens=True, truncation=True, padding="max_length")["input_ids"] for p in ents[s:s + 64]]
        e = m(torch.LongTensor(ids).to(dev))[1]
        e = e / torch.norm(e, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        out.append(e.cpu().numpy())
        if (s // 64) % 50 == 0:
            print(f'{s}/{len(ents)}', flush=True)
np.save(D / "all_entities_pretrain_emb.npy", np.concatenate(out))
print("embedded", len(ents))
