"""Focused checks for scripts/data/audit_assets.py on small temporary fixtures.

Failure list the audit must handle (each case below covers one or more):
malformed QA rows, duplicate question IDs, split overlap, lock mismatch, missing
OCR documents/pages, inventories without page_idx, alias collisions, evidence
page-index mismatch, PDF archive absent or present, symlink/hard-link double
counting, audit output re-counted, .env secret leakage, source mutation, and
non-deterministic manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("audit_assets", ROOT / "scripts/data/audit_assets.py")
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def no_configured_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in audit.PATH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _qa(qid: str, doc: str, page, context: str = "the quarterly revenue was 842 million") -> dict:
    return {
        "doc_name": doc,
        "ID": qid,
        "questions": f"question {qid}",
        "answers": "842",
        "doc_type": doc.split("/")[0],
        "answer_form": "Numeric",
        "evidence_source": "text",
        "evidence_context": context,
        "evidence_page_no": page,
    }


def make_project(tmp_path: Path, *, qas=None, splits=None, mineru_pages=None, link_src: bool = False) -> Path:
    root = tmp_path / "proj"
    base = root / "OHR-Bench/data/retrieval_base"
    pages = [{"page_idx": 0, "text": "cover page"}, {"page_idx": 1, "text": "The quarterly revenue was 842 million."}]
    _write(base / "gt/finance/docA.json", pages)
    _write(base / "gt/finance/docB.json", pages)
    _write(base / "MinerU/finance/docA.json", mineru_pages if mineru_pages is not None else pages)
    qas = qas if qas is not None else [_qa("q1", "finance/docA", 1), _qa("q2", "finance/docB", 1), _qa("q3", "finance/docA", 0, "cover page text")]
    _write(root / "OHR-Bench/data/qas_v2.json", qas)
    ids = [q["ID"] for q in qas if isinstance(q, dict) and "ID" in q]
    splits = splits if splits is not None else {"train": ids[:1], "val": ids[1:2], "test": ids[2:]}
    _write(root / "config/datasets/ohr_split.json", {"source": "OHR-Bench/data/qas_v2.json", "counts": {k: len(v) for k, v in splits.items()}, "splits": splits})
    _write(
        root / "config/split_checksums.json",
        {"split_sha256": _sha(root / "config/datasets/ohr_split.json"), "qas_v2_sha256": _sha(root / "OHR-Bench/data/qas_v2.json")},
    )
    (root / ".env").write_text("OPENAI_API_KEY=sk-secret-value-123\nHF_TOKEN=hf_secret456\n", encoding="utf-8")
    if link_src:
        (root / "data").mkdir()
        target = root / "OHR-Bench/data/qas_v2.json"
        os.symlink(target, root / "data/qas_link.json")
        os.link(target, root / "data/qas_hardlink.json")
    return root


def run(root: Path, *extra: str) -> tuple[int, dict]:
    out = root / "results/data_audit"
    code = audit.main(["--project-root", str(root), "--out-dir", str(out), *extra])
    return code, json.loads((out / "asset_manifest.json").read_text())


def codes(manifest: dict) -> set[str]:
    return {i["code"] for i in manifest["issues"]}


def test_clean_fixture_counts_and_pdf_gap(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    code, m = run(root)
    assert code == 0
    qa = m["ohr"]["qa"]
    assert (qa["records"], qa["unique_ids"], qa["raw_documents"]) == (3, 3, 2)
    cov = m["ohr"]["coverage"]["question_coverage"]
    assert cov["denominator_questions"] == 3
    assert cov["gt_evidence_ok"] == 3
    # docB has no MinerU file: missing assets are findings, not failures.
    assert cov["MinerU_status_counts"] == {"doc_missing": 1, "ok": 2}
    assert cov["pdf_status_counts"] == {"missing_source": 3}
    assert "pdf_source_missing" in codes(m)
    assert m["ohr"]["coverage"]["document_coverage"]["complete_document_coverage"]["status"] == "unknown"
    assert {c["result"] for c in m["hashes"]["lock_comparisons"]} == {"match"}
    # A completed audit can still report blockers.
    assert m["audit_execution"] == {"completed": True, "exit_code": 0, "integrity_failures": [], "meaning": m["audit_execution"]["meaning"]}
    assert m["experimental_readiness"]["status"] == "blocked"
    assert "pdf_source_missing" in m["experimental_readiness"]["blocking_issue_codes"]
    summary = (root / "results/data_audit/asset_audit_summary.md").read_text()
    assert "Experimental readiness: **blocked**" in summary


def test_missing_and_empty_ocr_stay_visible(tmp_path: Path) -> None:
    # docA MinerU lacks page 1 text (empty) ; docB has no MinerU file at all.
    root = make_project(tmp_path, mineru_pages=[{"page_idx": 0, "text": "cover"}, {"page_idx": 1, "text": "  "}])
    code, m = run(root)
    cov = m["ohr"]["coverage"]
    assert cov["question_coverage"]["denominator_questions"] == 3
    statuses = {q["id"]: q["MinerU_status"] for q in cov["per_question"]}
    assert statuses == {"q1": "page_empty", "q2": "doc_missing", "q3": "ok"}
    oc = cov["ocr_conditions"]["MinerU"]
    assert oc["documents_missing"] == ["finance/docB"]
    assert oc["documents_with_empty_pages"] == {"finance/docA": {"pages": [1], "gt_nonempty": [1]}}
    assert oc["page_counts"] == {"missing": 0, "empty": 1, "empty_where_gt_has_text": 1}
    assert oc["document_counts"]["with_any_condition"] == 2
    assert "no document or question is excluded" in cov["ocr_conditions"]["policy"]
    assert code == 0


def test_script_presence_flags(tmp_path: Path) -> None:
    qas = [_qa("q1", "finance/docA", 1), _qa("q2", "finance/docA", 1), _qa("q3", "finance/docA", 1), _qa("q4", "finance/docB", 1), _qa("q5", "finance/docB", 1)]
    for q, text in zip(qas, ["营收是多少", "売上はいくらですか", "매출은 얼마입니까", "東京の売上は", "What is the revenue?"]):
        q["questions"] = text
    root = make_project(tmp_path, qas=qas)
    _, m = run(root)
    scripts = m["ohr"]["qa"]["question_scripts"]
    assert scripts["questions_containing"] == {"han": 3, "kana": 2, "hangul": 1}
    assert scripts["script_combinations"] == {"han": 1, "han+kana": 2, "hangul": 1, "none": 1}
    assert scripts["questions_containing_any"] == 4
    assert "U+AC00-U+D7AF" in scripts["ranges_checked"]["hangul"]
    per_q = {q["id"]: q.get("question_scripts") for q in m["ohr"]["coverage"]["per_question"]}
    assert per_q["q4"] == ["han", "kana"] and per_q["q5"] is None


def test_canonically_equivalent_name_and_ambiguous_text_mapping(tmp_path: Path) -> None:
    import unicodedata

    root = make_project(tmp_path)
    pages = [{"page_idx": 0, "text": "The quarterly revenue was 842 million."}]
    nfc = unicodedata.normalize("NFC", "textbook/Saitô_book.pdf_12")
    _write(root / f"OHR-Bench/data/retrieval_base/gt/{nfc}.json", pages)
    # Two on-disk files both reachable from one QA name under the alias rules.
    _write(root / "OHR-Bench/data/retrieval_base/gt/textbook/jiaocai_needrop_en_7.json", pages)
    _write(root / "OHR-Bench/data/retrieval_base/gt/textbook/textbook_needrop_en_7.json", pages)
    qas = [_qa("q1", unicodedata.normalize("NFD", nfc), 0), _qa("q2", "textbook/textbook_needrop_en_7", 0)]
    _write(root / "OHR-Bench/data/qas_v2.json", qas)
    _write(root / "config/datasets/ohr_split.json", {"counts": {"train": 2}, "splits": {"train": ["q1", "q2"]}})
    (root / "config/split_checksums.json").unlink()
    _, m = run(root)
    gt = m["ohr"]["coverage"]["text_sources"]["gt"]
    res = {r["qa_doc_name"]: r for r in gt["alias_resolutions"]}
    nfd = unicodedata.normalize("NFD", nfc)
    assert res[nfd]["diagnosis"] == "unicode_normalisation_alias" and res[nfd]["resolved_key"] == nfc
    assert gt["ambiguous_mappings"] == {"textbook/textbook_needrop_en_7": ["textbook/jiaocai_needrop_en_7", "textbook/textbook_needrop_en_7"]}
    assert "ambiguous_mapping" in codes(m)


def test_malformed_and_duplicate_qa_exit_2(tmp_path: Path) -> None:
    bad = [_qa("q1", "finance/docA", 1), _qa("q1", "finance/docB", 1), {"doc_name": "finance/docA", "ID": "q9"}, _qa("q4", "finance/docA", "one")]
    root = make_project(tmp_path, qas=bad, splits={"train": ["q1"], "val": ["q9"], "test": ["q4"]})
    code, m = run(root)
    assert code == 2
    assert m["ohr"]["qa"]["duplicate_ids"] == ["q1"]
    assert len(m["ohr"]["qa"]["malformed_records"]) == 2
    assert {"qa_duplicate_ids", "qa_malformed_rows"} <= codes(m)


def test_duplicate_ids_alone_exit_2(tmp_path: Path) -> None:
    dup = [_qa("q1", "finance/docA", 1), _qa("q1", "finance/docB", 1)]
    root = make_project(tmp_path, qas=dup, splits={"train": ["q1"]})
    code, m = run(root)
    assert code == 2
    assert next(i for i in m["issues"] if i["code"] == "qa_duplicate_ids")["severity"] == "integrity"


def test_split_overlap_and_lock_mismatch_exit_2(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    split_path = root / "config/datasets/ohr_split.json"
    payload = json.loads(split_path.read_text())
    payload["splits"]["test"].append("q1")
    payload["counts"]["test"] += 1
    split_path.write_text(json.dumps(payload))
    code, m = run(root)
    assert code == 2
    assert m["ohr"]["split"]["question_id_overlap"]["test&train"] == 1
    assert {"split_id_overlap", "lock_mismatch"} <= codes(m)


def test_page_index_mismatch_and_missing_page_idx(tmp_path: Path) -> None:
    # Evidence page 2 does not exist; its context sits on page 1 (a 1-based signal).
    qas = [_qa("q1", "finance/docA", 2), _qa("q2", "finance/docB", 1)]
    mineru = [{"text": "no page index here"}, {"page_idx": 1, "text": ""}]
    root = make_project(tmp_path, qas=qas, mineru_pages=mineru)
    code, m = run(root)
    assert code == 0
    cov = m["ohr"]["coverage"]
    q1 = next(q for q in cov["per_question"] if q["id"] == "q1")
    assert q1["gt_status"] == "page_missing" and q1["gt_missing_pages"] == [2]
    assert q1["alignment"] == "on_page_minus_1"
    assert cov["page_convention"]["evidence_page_out_of_gt_range"] == 1
    # A missing page_idx is reported, never defaulted to page 0.
    assert cov["text_sources"]["MinerU"]["malformed_files"] == {"finance/docA": ["1 entries missing page_idx"]}
    assert cov["text_sources"]["MinerU"]["empty_pages_total"] == 1


def test_alias_resolution_and_collision(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    pages = [{"page_idx": 0, "text": "The quarterly revenue was 842 million."}]
    _write(root / "OHR-Bench/data/retrieval_base/gt/textbook/jiaocai_needrop_en_7.json", pages)
    qas = [_qa("q1", "textbook/textbook_needrop_en_7", 0), _qa("q2", "textbook/jiaocai_needrop_en_7", 0)]
    _write(root / "OHR-Bench/data/qas_v2.json", qas)
    _write(root / "config/datasets/ohr_split.json", {"counts": {"train": 1, "test": 1}, "splits": {"train": ["q1"], "test": ["q2"]}})
    (root / "config/split_checksums.json").unlink()
    code, m = run(root)
    gt = m["ohr"]["coverage"]["text_sources"]["gt"]
    assert gt["documents_resolved"]["alias"] == 1
    assert gt["alias_collisions"] == {"textbook/jiaocai_needrop_en_7": ["textbook/jiaocai_needrop_en_7", "textbook/textbook_needrop_en_7"]}
    assert "alias_collision" in codes(m)
    overlap = m["ohr"]["coverage"]["split_documents"]["pairwise_overlap"]["test&train"]
    # Distinct raw names overlap only after the documented alias rule.
    assert (overlap["raw_document_names"], overlap["alias_normalised_documents"]) == (0, 1)
    assert code == 0


def test_pdf_archive_is_read_without_extraction(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    archive = root / "data/ohr_bench_raw/pdfs.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("finance/docA.pdf", b"%PDF-1.4 fake")
    code, m = run(root)
    cov = m["ohr"]["coverage"]
    # The member is read but is not a valid PDF: a genuinely unreadable PDF, not a missing parser.
    expected = "unreadable_pdf" if audit.pdf_parser_available() else "parser_unavailable"
    assert cov["pdf_coverage"]["status_counts"] == {expected: 1, "missing_source": 1}
    loc = next(loc for loc in cov["pdf_sources"]["locations"] if loc["kind"] == "pdf_zip")
    assert loc["pdf_members"] == 1 and loc["exists"]
    assert sorted(p.name for p in archive.parent.iterdir()) == ["pdfs.zip"]
    assert "pdf_source_missing" not in codes(m)
    assert code == 0


def _pdf(pages: int) -> bytes:
    """Minimal valid PDF with `pages` blank pages and a correct xref table."""
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>"]
    objs += ["<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>"] * pages
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_complete_document_page_counts_from_pdf_root(tmp_path: Path) -> None:
    pytest.importorskip("pypdfium2")
    root = make_project(tmp_path)
    pdf_root = tmp_path / "pdfs/finance"
    pdf_root.mkdir(parents=True)
    (pdf_root / "docA.pdf").write_bytes(_pdf(2))  # matches gt pages 0..1
    (pdf_root / "docB.pdf").write_bytes(_pdf(3))  # gt lacks page 2
    code, m = run(root, "--pdf-root", str(tmp_path / "pdfs"))
    complete = m["ohr"]["coverage"]["document_coverage"]["complete_document_coverage"]
    assert complete["status"] == "checked"
    assert (complete["numerator_verified_complete"], complete["denominator_documents"]) == (1, 2)
    pc = m["ohr"]["coverage"]["pdf_coverage"]
    assert pc["page_set_mismatches"] == [{"doc_name": "finance/docB", "pdf_pages": 3, "gt_pages_missing_vs_pdf": [2], "gt_pages_beyond_pdf": []}]
    # docB has no MinerU file; docA MinerU covers every PDF page.
    assert pc["mineru_pages_missing_vs_pdf"] == {}
    per_doc = m["ohr"]["coverage"]["pdf_per_document"]
    assert per_doc["finance/docA"]["pdf_sha256"] == hashlib.sha256(_pdf(2)).hexdigest()
    assert "pdf_page_set_mismatch" in codes(m)
    assert code == 0


def _zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def test_corrupt_zip_does_not_block_directory_pdfs(tmp_path: Path) -> None:
    pytest.importorskip("pypdfium2")
    root = make_project(tmp_path)
    bad = root / "data/ohr_bench_raw/pdfs.zip"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"PK\x03\x04 this is not a complete archive")
    (tmp_path / "pdfs/finance").mkdir(parents=True)
    (tmp_path / "pdfs/finance/docA.pdf").write_bytes(_pdf(2))
    code, m = run(root, "--pdf-root", str(tmp_path / "pdfs"))
    pc = m["ohr"]["coverage"]["pdf_coverage"]
    assert pc["status_counts"] == {"missing_source": 1, "verified_complete": 1}
    assert {"pdf_archive_unreadable", "pdf_document_missing"} <= codes(m)
    assert code == 0


def test_damaged_member_and_unreadable_file_leave_others_auditable(tmp_path: Path) -> None:
    pytest.importorskip("pypdfium2")
    root = make_project(tmp_path)
    archive = root / "data/ohr_bench_raw/pdfs.zip"
    _zip(archive, {"finance/docA.pdf": _pdf(2), "finance/docB.pdf": _pdf(2) * 40})
    raw = bytearray(archive.read_bytes())
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("finance/docB.pdf")
    start = info.header_offset + 30 + len(info.filename) + len(info.extra)
    raw[start + 5:start + 40] = b"\xff" * 35  # damage docB's compressed stream only
    archive.write_bytes(bytes(raw))
    code, m = run(root)
    per_doc = m["ohr"]["coverage"]["pdf_per_document"]
    assert per_doc["finance/docA"]["status"] == "verified_complete"
    assert per_doc["finance/docB"]["status"] == "unreadable_source"
    assert per_doc["finance/docB"]["reason"].startswith("archive_member_corrupt")
    assert "pdf_source_unreadable" in codes(m) and code == 0


def test_encrypted_member_and_unreadable_directory_file(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    archive = root / "data/ohr_bench_raw/pdfs.zip"
    _zip(archive, {"finance/docA.pdf": _pdf(2)})
    raw = bytearray(archive.read_bytes())
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("finance/docA.pdf")
    raw[info.header_offset + 6] |= 0x1  # local header: encrypted flag
    central = raw.rfind(b"PK\x01\x02")
    raw[central + 8] |= 0x1  # central directory: encrypted flag
    archive.write_bytes(bytes(raw))
    pdf_dir = tmp_path / "pdfs/finance"
    pdf_dir.mkdir(parents=True)
    unreadable = pdf_dir / "docB.pdf"
    unreadable.write_bytes(_pdf(2))
    unreadable.chmod(0)
    try:
        code, m = run(root, "--pdf-root", str(tmp_path / "pdfs"))
    finally:
        unreadable.chmod(0o644)
    per_doc = m["ohr"]["coverage"]["pdf_per_document"]
    assert per_doc["finance/docA"]["reason"] == "encrypted_archive_member_unsupported"
    if os.geteuid() != 0:
        assert per_doc["finance/docB"]["reason"].startswith("source_file_unreadable")
    assert code == 0


def test_missing_parser_differs_from_unreadable_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path)
    _zip(root / "data/ohr_bench_raw/pdfs.zip", {"finance/docA.pdf": _pdf(2), "finance/docB.pdf": b"%PDF-1.4 broken"})
    monkeypatch.setattr(audit, "pdf_parser_available", lambda: False)
    code, m = run(root)
    pc = m["ohr"]["coverage"]["pdf_coverage"]
    assert pc["status_counts"] == {"parser_unavailable": 2}
    assert m["ohr"]["coverage"]["document_coverage"]["complete_document_coverage"]["status"] == "unknown"
    assert "pdf_parser_unavailable" in m["experimental_readiness"]["blocking_issue_codes"]
    assert code == 0
    monkeypatch.undo()
    pytest.importorskip("pypdfium2")
    _, m = run(root)
    assert m["ohr"]["coverage"]["pdf_coverage"]["status_counts"] == {"unreadable_pdf": 1, "verified_complete": 1}


def test_conflicting_equivalent_members_are_ambiguous(tmp_path: Path) -> None:
    import unicodedata

    root = make_project(tmp_path)
    name = "finance/docé"
    pages = [{"page_idx": 0, "text": "x"}, {"page_idx": 1, "text": "The quarterly revenue was 842 million."}]
    _write(root / f"OHR-Bench/data/retrieval_base/gt/{unicodedata.normalize('NFC', name)}.json", pages)
    qas = [_qa("q1", unicodedata.normalize("NFC", name), 1)]
    _write(root / "OHR-Bench/data/qas_v2.json", qas)
    _write(root / "config/datasets/ohr_split.json", {"counts": {"train": 1}, "splits": {"train": ["q1"]}})
    (root / "config/split_checksums.json").unlink()
    _zip(
        root / "data/ohr_bench_raw/pdfs.zip",
        {unicodedata.normalize("NFC", name) + ".pdf": _pdf(2), unicodedata.normalize("NFD", name) + ".pdf": _pdf(3)},
    )
    _, m = run(root)
    doc = m["ohr"]["coverage"]["pdf_per_document"][unicodedata.normalize("NFC", name)]
    assert doc["status"] == "ambiguous_source" and len(doc["candidates"]) == 2
    assert "selected" not in doc
    assert "pdf_source_ambiguous" in codes(m)


def test_links_output_and_secrets(tmp_path: Path) -> None:
    root = make_project(tmp_path, link_src=True)
    run(root)
    code, m = run(root)  # second run must not count the first run's output
    roots = {r["label"]: r for r in m["inventory"]["roots"]}
    data = roots["data"]
    assert data["symlinks_not_followed"] == ["data/qas_link.json"]
    assert data["files"] == 0
    assert data["hardlink_duplicates_not_counted"][0]["path"] == "data/qas_hardlink.json"
    assert roots["results"]["files"] == 0 and roots["results"]["excluded"] == {"audit_output": 1}
    text = (root / "results/data_audit/asset_manifest.json").read_text()
    assert "sk-secret" not in text and "hf_secret" not in text
    assert code == 0


def test_repeatable_and_read_only(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    sources = sorted(p for p in root.rglob("*") if p.is_file())
    before = {p: _sha(p) for p in sources}
    _, first = run(root)
    _, second = run(root)
    assert first["content_sha256"] == second["content_sha256"]
    for manifest in (first, second):
        for key in ("generated_at_utc", "runtime_sec", "content_sha256"):
            manifest.pop(key)
    assert first == second
    assert {p: _sha(p) for p in sources} == before


def test_invalid_project_root_exit_1(tmp_path: Path) -> None:
    assert audit.main(["--project-root", str(tmp_path / "missing"), "--out-dir", str(tmp_path / "out")]) == 1


def test_reader_reports_archive_that_turns_unreadable_after_listing(tmp_path: Path) -> None:
    archive = tmp_path / "pdfs.zip"
    archive.write_bytes(b"not a zip")
    reader = audit.PdfReader()
    entry = {"kind": "pdf_zip", "member": "finance/docA.pdf", "_zip": archive}
    assert reader.read(entry) == (None, "archive_unreadable: BadZipFile")
    assert reader.read({"kind": "pdf_root", "_abs": tmp_path / "gone.pdf"}) == (None, "source_file_missing")
    reader.close()


def test_flat_tree_hash_matches_git(tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    (repo / "cat").mkdir(parents=True)
    (repo / "cat/a.json").write_text("[]\n")
    (repo / "cat/b.json").write_text('[{"page_idx": 0}]\n')
    (repo / "cat/b.json").chmod(0o755)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)
    want = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD:cat"], capture_output=True, text=True, check=True).stdout.strip()
    assert audit.git_flat_tree_sha1(repo / "cat") == want
    assert audit.git_blob_sha1(repo / "cat/a.json") == subprocess.run(["git", "hash-object", str(repo / "cat/a.json")], capture_output=True, text=True, check=True).stdout.strip()
