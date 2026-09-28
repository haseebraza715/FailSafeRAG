"""Focused checks for scripts/data/build_pilot.py on a small end-to-end fixture.

Each test runs the real asset audit and then the pilot builder on a temporary
project, so the checks exercise the same inputs the real command reads.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = spec_from_file_location("build_pilot", ROOT / "scripts/data/build_pilot.py")
pilot = module_from_spec(_spec)
_spec.loader.exec_module(pilot)
audit = pilot.audit

pytest.importorskip("pypdfium2")
pytest.importorskip("PIL")


@pytest.fixture(autouse=True)
def no_configured_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in audit.PATH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _pdf(pages: int) -> bytes:
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>"]
    objs += ["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 60 80] >>"] * pages
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# docs -> (MinerU pages or None for a missing file)
DOCS = {
    "finance/f1": [{"page_idx": 0, "text": "revenue table"}],  # page 1 missing
    "finance/f2": [{"page_idx": 0, "text": "a"}, {"page_idx": 1, "text": "b"}],
    "news/n1": [{"page_idx": 0, "text": "   "}, {"page_idx": 1, "text": "<b>bold</b> & more"}],  # page 0 empty
    "news/n2": [{"page_idx": 0, "text": "a"}, {"page_idx": 1, "text": "b"}],
    "textbook/book.pdf_5": None,  # no MinerU file at all
    "finance/shared": [{"page_idx": 0, "text": "a"}, {"page_idx": 1, "text": "b"}],
    "news/testonly": [{"page_idx": 0, "text": "a"}, {"page_idx": 1, "text": "b"}],
    "textbook/book.pdf_9": [{"page_idx": 0, "text": "a"}, {"page_idx": 1, "text": "b"}],
}


def _q(qid: str, doc: str, page, text: str | None = None) -> dict:
    return {
        "doc_name": doc, "ID": qid, "questions": text or f"What is in {qid}?", "answers": f"ANS-{qid}",
        "doc_type": doc.split("/")[0], "answer_form": "Short", "evidence_source": "text",
        "evidence_context": f"CTX-{qid} gold evidence", "evidence_page_no": page,
    }


def make_project(tmp_path: Path, *, reverse: bool = False, pdf_missing: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "proj"
    base = root / "OHR-Bench/data/retrieval_base"
    for doc, mineru in DOCS.items():
        _write(base / f"gt/{doc}.json", [{"page_idx": 0, "text": f"GTTEXT {doc} p0"}, {"page_idx": 1, "text": f"GTTEXT {doc} p1"}])
        if mineru is not None:
            _write(base / f"MinerU/{doc}.json", mineru)
    qas = [
        _q("a1", "finance/f1", 1), _q("a2", "finance/f1", [0, 1]),
        _q("b1", "finance/f2", 0),
        _q("c1", "news/n1", 0, "<script>alert(1)</script> 营收?"), _q("c2", "news/n1", 1),
        _q("d1", "news/n2", 1),
        _q("e1", "textbook/book.pdf_5", 0),
        _q("s1", "finance/shared", 0), _q("s2", "finance/shared", 1),
        _q("t1", "news/testonly", 0),
        _q("v1", "textbook/book.pdf_9", 0),
    ]
    splits = {"train": ["a1", "a2", "b1", "c1", "c2", "d1", "e1", "s1"], "val": ["s2", "v1"], "test": ["t1"]}
    if reverse:
        qas = qas[::-1]
        splits = {k: v[::-1] for k, v in reversed(list(splits.items()))}
    _write(root / "OHR-Bench/data/qas_v2.json", qas)
    _write(root / "config/datasets/ohr_split.json", {"counts": {k: len(v) for k, v in splits.items()}, "splits": splits})
    _write(root / "config/split_checksums.json", {"split_sha256": _sha(root / "config/datasets/ohr_split.json"), "qas_v2_sha256": _sha(root / "OHR-Bench/data/qas_v2.json")})
    archive = root / "data/ohr_bench_raw/pdfs.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        for doc in DOCS:
            if doc not in pdf_missing:
                zf.writestr(f"{doc}.pdf", _pdf(2))
    _write(root / "config/ohr_pdf_source_lock.json", {"pdf_archive": {"local_path": "data/ohr_bench_raw/pdfs.zip", "sha256": _sha(archive)}, "qa_and_text_reference": {}})
    return root


def write_config(root: Path, *, documents: int = 3, seed: int = 42, cases: int = 4, budget: int = 3, name: str = "test_pilot") -> Path:
    config = {
        "schema_version": 1, "pilot_id": name, "dataset": "OHR-Bench",
        "qa_source": "OHR-Bench/data/qas_v2.json", "split": "config/datasets/ohr_split.json",
        "audit_manifest": "results/data_audit/asset_manifest.json", "pdf_source_lock": "config/ohr_pdf_source_lock.json",
        "noisy_text_source": "MinerU", "reference_text_source": "gt", "output_dir": f"results/pilots/{name}",
        "selection": {"algorithm": "stratified-hash-rank-v1", "seed": seed, "documents": documents, "pool": "train_exclusive", "stratify_by": "doc_type", "minimum_per_available_stratum": 1},
        "inspection": {"algorithm": "purposive-quota-hash-rank-v1", "cases": cases, "page_budget": budget, "dpi": 30, "image_format": "png",
                       "category_quotas": [["ocr_gap", 2], ["multi_page_evidence", 1], ["han_script", 1], ["text", 1]]},
    }
    path = root / f"config/pilots/{name}.json"
    _write(path, config)
    return path


def run_audit(root: Path) -> None:
    code = audit.main(["--project-root", str(root), "--out-dir", str(root / "results/data_audit"), "--pdf-zip", str(root / "data/ohr_bench_raw/pdfs.zip")])
    assert code == 0


def build(root: Path, config: Path, *extra: str) -> int:
    return pilot.main(["--project-root", str(root), "--config", str(config), *extra])


def load(root: Path, name: str = "test_pilot") -> tuple[dict, dict, dict]:
    out = root / f"results/pilots/{name}"
    return tuple(json.loads((out / f).read_text(encoding="utf-8")) for f in ("selection_record.json", "runtime_manifest.json", "evaluation_manifest.json"))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = make_project(tmp_path)
    run_audit(root)
    return root


def test_selection_is_train_exclusive_and_questions_are_train(project: Path) -> None:
    assert build(project, write_config(project)) == 0
    record, runtime, _ = load(project)
    selected = {d["doc_id"] for d in record["selected_documents"]}
    assert record["pool"]["train_exclusive_documents"] == 5  # f1, f2, n1, n2, book.pdf_5
    assert not selected & {"finance/shared", "news/testonly", "textbook/book.pdf_9"}
    split = json.loads((project / "config/datasets/ohr_split.json").read_text())["splits"]
    assert {q["question_id"] for q in runtime["questions"]} <= set(split["train"])
    # every stratum with eligible documents gets a slot when feasible
    assert {r["doc_type"]: r["allocated"] for r in record["allocation"]} == {"finance": 1, "news": 1, "textbook": 1}


def test_within_stratum_choice_follows_seeded_rank(project: Path) -> None:
    assert build(project, write_config(project, seed=42)) == 0
    record, _, _ = load(project)
    eligible = ["finance/f1", "finance/f2", "news/n1", "news/n2", "textbook/book.pdf_5"]
    for row in record["selected_documents"]:
        stratum = [d for d in eligible if d.split("/")[0] == row["doc_type"]]
        assert row["rank_key"] == pilot.rank_key(42, "select", row["doc_id"])
        assert row["doc_id"] == min(stratum, key=lambda d: pilot.rank_key(42, "select", d))


def test_selection_is_independent_of_input_order(tmp_path: Path) -> None:
    a = make_project(tmp_path / "a")
    b = make_project(tmp_path / "b", reverse=True)
    for root in (a, b):
        run_audit(root)
        assert build(root, write_config(root)) == 0
    ra, rta, eva = load(a)
    rb, rtb, evb = load(b)
    assert ra["selected_documents"] == rb["selected_documents"]
    assert ra["selected_question_ids"] == rb["selected_question_ids"]
    assert ra["manifest_digests"] == rb["manifest_digests"]
    assert pilot.digest(rta) == pilot.digest(rtb) and pilot.digest(eva) == pilot.digest(evb)


def test_frozen_pilot_is_reproduced_or_refused(project: Path) -> None:
    config = write_config(project)
    assert build(project, config) == 0
    first = (project / "results/pilots/test_pilot/selection_record.json").read_text()
    assert build(project, config) == 0  # same inputs: reproduced, record untouched
    assert (project / "results/pilots/test_pilot/selection_record.json").read_text() == first
    write_config(project, seed=7)  # same pilot_id/output_dir, different selection config
    assert build(project, config, "--no-render") == pilot.EXIT_FROZEN_MISMATCH  # the selection record guard alone
    assert (project / "results/pilots/test_pilot/selection_record.json").read_text() == first
    write_config(project)
    runtime = project / "results/pilots/test_pilot/runtime_manifest.json"
    runtime.write_text(runtime.read_text().replace('"ok"', '"missing"', 1))
    assert build(project, config) == pilot.EXIT_FROZEN_MISMATCH


def test_runtime_manifest_excludes_gold_and_gt(project: Path) -> None:
    assert build(project, write_config(project, documents=5)) == 0
    _, runtime, evaluation = load(project)
    text = json.dumps(runtime, ensure_ascii=False)
    for forbidden in ("ANS-", "CTX-", "GTTEXT", "retrieval_base/gt", "evidence_page_no", "evidence_context", '"answers"'):
        assert forbidden not in text, forbidden
    ev_text = json.dumps(evaluation, ensure_ascii=False)
    assert "ANS-a1" in ev_text and "CTX-a1" in ev_text and "retrieval_base/gt" in ev_text


def test_complete_page_inventory_keeps_ocr_gaps(project: Path) -> None:
    assert build(project, write_config(project, documents=5)) == 0
    _, runtime, evaluation = load(project)
    docs = {d["doc_id"]: d for d in runtime["documents"]}
    assert [p["ocr_status"] for p in docs["finance/f1"]["pages"]] == ["ok", "missing"]
    assert [p["ocr_status"] for p in docs["news/n1"]["pages"]] == ["empty", "ok"]
    book = docs["textbook/book.pdf_5"]
    assert book["noisy_text"]["status"] == "missing" and [p["ocr_status"] for p in book["pages"]] == ["missing", "missing"]
    assert all(len(d["pages"]) == d["pdf"]["page_count"] == 2 for d in docs.values())
    assert evaluation["questions"]["a1"]["evidence_page_ocr_status"] == "missing"
    assert evaluation["questions"]["a2"]["list_valued_evidence"] and evaluation["questions"]["a2"]["multi_page_evidence"]
    assert evaluation["questions"]["c1"]["question_scripts"] == ["han"]


def test_unusable_source_is_an_explicit_exclusion_and_shortfall_stops(tmp_path: Path) -> None:
    root = make_project(tmp_path, pdf_missing=("news/n2",))
    run_audit(root)
    assert build(root, write_config(root, documents=4)) == 0
    record, _, _ = load(root)
    exclusions = {e["doc_id"]: e["reasons"] for e in record["pool"]["source_exclusions"]}
    assert exclusions == {"news/n2": ["source PDF status ['missing_source']"]}
    assert record["pool"]["eligible_documents"] == 4
    config = write_config(root, documents=5, name="too_big")
    assert build(root, config) == pilot.EXIT_INVALID
    assert not (root / "results/pilots/too_big").exists()


def test_hashes_match_selected_inputs(project: Path) -> None:
    assert build(project, write_config(project, documents=5)) == 0
    _, runtime, _ = load(project)
    with zipfile.ZipFile(project / "data/ohr_bench_raw/pdfs.zip") as zf:
        for d in runtime["documents"]:
            assert d["pdf"]["sha256"] == hashlib.sha256(zf.read(d["pdf"]["member"])).hexdigest()
            if d["noisy_text"]["path"]:
                path = project / d["noisy_text"]["path"]
                assert d["noisy_text"]["sha256"] == _sha(path)
                pages = {p["page_idx"]: p["text"] for p in json.loads(path.read_text())}
                for page in d["pages"]:
                    if page["ocr_text_sha256"]:
                        assert page["ocr_text_sha256"] == hashlib.sha256(pages[page["page_idx"]].encode()).hexdigest()


def test_render_respects_budget_and_records_provenance(project: Path) -> None:
    assert build(project, write_config(project, documents=5, cases=6, budget=3)) == 0
    out = project / "results/pilots/test_pilot/inspection"
    record = json.loads((out / "render_record.json").read_text())
    cases = json.loads((out / "inspection_cases.json").read_text())
    assert record["unique_pages"] == len(cases["unique_pages"]) <= 3
    assert record["renderer"]["package"] == "pypdfium2" and record["dpi"] == 30
    for page in record["pages"]:
        assert (out / page["file"]).is_file() and page["png_sha256"] == _sha(out / page["file"])
        assert page["source_sha256"] and page["width"] > 0 and page["height"] > 0
    assert len(cases["cases"]) < 6 or cases["skipped_for_page_budget"] == []


def test_html_escapes_text_and_uses_local_assets(project: Path) -> None:
    assert build(project, write_config(project, documents=5, cases=7, budget=20)) == 0  # every question is a case
    out = project / "results/pilots/test_pilot/inspection"
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "<script" not in page.lower().replace("&lt;script", "")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; more" in page
    assert "http://" not in page and "https://" not in page
    for src in __import__("re").findall(r"src='([^']+)'", page):
        assert (out / src).is_file()
    assert "Inspection only" in page and "entry present but empty" in page


def test_annotations_are_never_overwritten(project: Path) -> None:
    config = write_config(project, documents=5)
    assert build(project, config) == 0
    csv_path = project / "results/pilots/test_pilot/inspection/annotations.csv"
    csv_path.write_text(csv_path.read_text() + "human note\n")
    assert build(project, config) == 0
    assert csv_path.read_text().endswith("human note\n")


def test_sources_unchanged_and_stale_audit_rejected(project: Path) -> None:
    sources = [p for p in project.rglob("*") if p.is_file() and "results" not in p.parts and "pilots" not in p.parts]
    before = {p: _sha(p) for p in sources}
    assert build(project, write_config(project)) == 0
    assert {p: _sha(p) for p in sources} == before
    mineru = project / "OHR-Bench/data/retrieval_base/MinerU/news/n2.json"
    mineru.write_text(json.dumps([{"page_idx": 0, "text": "changed"}]))
    assert build(project, write_config(project, name="after_change")) == pilot.EXIT_INVALID
