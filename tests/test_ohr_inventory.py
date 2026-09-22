from __future__ import annotations

import json
from pathlib import Path

from faar.ohr_inventory import diagnose_ohr_inventory_gaps, resolve_ohr_inventory_path


def test_resolves_textbook_needrop_alias(tmp_path: Path) -> None:
    gt = tmp_path / "gt/textbook"
    gt.mkdir(parents=True)
    (gt / "jiaocai_needrop_en_75.json").write_text(json.dumps([{"text": "page", "page_idx": 0}]))

    path, resolved, kind = resolve_ohr_inventory_path(
        tmp_path / "gt",
        "textbook/textbook_needrop_en_75",
    )

    assert kind == "alias"
    assert resolved == "textbook/jiaocai_needrop_en_75"
    assert path is not None and path.is_file()


def test_unicode_mojibake_names_share_skeleton_and_resolve(tmp_path: Path) -> None:
    from faar.ohr_inventory import _skeleton

    gt = tmp_path / "gt/textbook"
    gt.mkdir(parents=True)
    # Mirror the real OHR mismatch: gt uses U+00C3 U+00B4, qas uses A + combining tilde + acute.
    on_disk = (
        "(Springer_Monographs_in_Mathematics)_Kazuyuki_Sait"
        + "\u00c3\u00b4"
        + ",_J._D._Maitland_Wright_-_Monotone_Complete_C_-algebras_and_Generic_Dynamics-Springer_(2015).pdf_122"
    )
    (gt / f"{on_disk}.json").write_text(json.dumps([{"text": "page", "page_idx": 0}]))
    qas_name = (
        "textbook/(Springer_Monographs_in_Mathematics)_Kazuyuki_Sait"
        + "A\u0303\u00b4"
        + ",_J._D._Maitland_Wright_-_Monotone_Complete_C_-algebras_and_Generic_Dynamics-Springer_(2015).pdf_122"
    )

    assert _skeleton(Path(qas_name).name) == _skeleton(on_disk)
    path, resolved, kind = resolve_ohr_inventory_path(tmp_path / "gt", qas_name)
    # APFS may treat the two spellings as the same filename ("exact"); either way the inventory loads.
    assert kind in {"exact", "unicode_alias"}
    assert path is not None and path.is_file()
    assert resolved is not None



def test_diagnose_separates_unavailable(tmp_path: Path) -> None:
    qas = tmp_path / "qas.json"
    gt = tmp_path / "gt"
    (gt / "textbook").mkdir(parents=True)
    (gt / "textbook/jiaocai_needrop_en_75.json").write_text(json.dumps([{"page_idx": 0}]))
    qas.write_text(
        json.dumps(
            [
                {"doc_name": "textbook/textbook_needrop_en_75", "ID": "1"},
                {"doc_name": "textbook/missing_doc", "ID": "2"},
            ]
        )
    )

    report = diagnose_ohr_inventory_gaps(
        qas_path=qas,
        inventory_dir=gt,
        pdf_names={"textbook/jiaocai_needrop_en_75.pdf"},
    )

    assert len(report["naming_mismatches"]) == 1
    assert report["naming_mismatches"][0]["page_count"] == 1
    assert report["unavailable"][0]["qas_doc_name"] == "textbook/missing_doc"
