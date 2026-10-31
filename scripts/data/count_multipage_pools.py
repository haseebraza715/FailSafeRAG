"""Count multi-page questions in candidate pools for a longer-document sample. Metadata only; writes nothing.

    PYTHONPATH=src .local/venv-prebaseline/bin/python scripts/data/count_multipage_pools.py

Reads the locked QA file, the fixed split, the asset audit, MinerU page inventories (for empty-page status) and the
frozen selections of ohr_dev_v1 and ohr_dev_v2. Never reads answers, evidence text or model outputs. Eligibility
follows build_pilot.py: one raw QA name, one verified source PDF and a gt reference.
"""
import json
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = spec_from_file_location("bp", root / "scripts/data/build_pilot.py")
bp = module_from_spec(spec)
spec.loader.exec_module(bp)
config = json.loads((root / "config/pilots/ohr_dev_v1.json").read_text())
inputs = bp.load_inputs(root, config)
pool = bp.build_pool(inputs)
manifest = inputs["manifest"]
pdf_docs = manifest["ohr"]["coverage"]["pdf_per_document"]
gt_unresolved = set(manifest["ohr"]["coverage"]["text_sources"]["gt"]["unresolved_documents"])
def ok(cid):
    raws = pool["raw_by_doc"][cid]
    if len(raws) > 1:
        return False
    st = pdf_docs.get(raws[0], {})
    return st.get("status") == "verified_complete" and len(st.get("candidates", [])) == 1 and raws[0] not in gt_unresolved
used = set()
for p in ("ohr_dev_v1", "ohr_dev_v2"):
    used |= {d["doc_id"] for d in json.loads((root / f"results/pilots/{p}/selection_record.json").read_text())["selected_documents"]}
all_docs = [d for d in pool["splits_by_doc"] if ok(d)]
prof_pool = {**pool, "eligible": all_docs}
pages, ocr, qflags = bp.pool_profiles(root, inputs, prof_pool, config)
split_of = pool["split_of"]
sbd = {d: set(s) for d, s in pool["splits_by_doc"].items()}
def describe(name, qs):
    mp = [q for q in qs if pages[q["doc_id"]] > 1]
    docs = {q["doc_id"] for q in mp}
    return {"pool": name, "questions": len(qs), "documents": len({q["doc_id"] for q in qs}),
            "questions_on_multi_page_docs": len(mp), "multi_page_docs": len(docs),
            "on_docs_6plus_pages": sum(pages[q["doc_id"]] >= 6 for q in qs),
            "multi_page_and_evidence_ocr_gap": sum(q["evidence_page_ocr_status"] != "ok" for q in mp),
            "multi_page_and_doc_has_ocr_gap": sum(ocr[q["doc_id"]]["empty"] + ocr[q["doc_id"]]["missing"] > 0 for q in mp),
            "multi_page_evidence": sum(q["multi_page_evidence"] for q in qs),
            "q_per_multi_page_doc": round(len(mp) / len(docs), 2) if docs else None,
            "multi_page_by_type": dict(sorted(Counter(bp.doc_type(q["doc_id"]) for q in mp).items()))}
train_q = [q for q in qflags if split_of.get(q["question_id"]) == "train"]
rows = [
    describe("A train-only documents, outside ohr_dev_v1 and ohr_dev_v2", [q for q in train_q if sbd[q["doc_id"]] == {"train"} and q["doc_id"] not in used]),
    describe("B train questions on documents shared with validation or test", [q for q in train_q if sbd[q["doc_id"]] != {"train"}]),
    describe("C validation questions (all documents)", [q for q in qflags if split_of.get(q["question_id"]) == "val"]),
    describe("E train questions on documents shared with validation only (no test question)", [q for q in train_q if sbd[q["doc_id"]] == {"train", "val"}]),
    describe("F train questions on documents with no test question (train-only or train+val)", [q for q in train_q if "test" not in sbd[q["doc_id"]] and q["doc_id"] not in used]),
    describe("D train-only documents, all (for reference)", [q for q in train_q if sbd[q["doc_id"]] == {"train"}]),
]
option_f_docs = {q["doc_id"] for q in train_q if "test" not in sbd[q["doc_id"]] and q["doc_id"] not in used and pages[q["doc_id"]] > 1}
rows.append(describe("G validation questions on multi-page documents outside option F (confirmation pool)",
                     [q for q in qflags if split_of.get(q["question_id"]) == "val" and pages[q["doc_id"]] > 1 and q["doc_id"] not in option_f_docs]))
print(json.dumps({"eligible_docs_all_splits": len(all_docs), "split_names": sorted(set(split_of.values())), "rows": rows}, indent=1))
