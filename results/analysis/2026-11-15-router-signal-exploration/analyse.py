"""Router signal exploration (offline). Rerun from the repo root (Code/):
    PYTHONPATH=src .local/venv-prebaseline/bin/python results/analysis/2026-11-15-router-signal-exploration/analyse.py
Reads existing result files only. Features are runtime-only: nothing from evaluation_manifest.json, no gold
answers, no evidence pages. The EM columns are used as labels/targets and for policy scoring only.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
SEED = 20261115
N_BOOT = 2000
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DEV, ANA, PIL = ROOT / "results/development", ROOT / "results/analysis", ROOT / "results/pilots"
P = "2026-11-01-ohr-"
SAMPLES = {
    "practice": dict(pilot="ohr_long_dev_v1", cases=P + "long-dev-v1-round-2", diag=P + "long-dev-v1-r1-diagnosis",
                     base=P + "long-dev-v1-baseline-answer-v2-terra-r1",
                     runs=dict(retry_retrieval=P + "long-dev-v1-r2-retry-keep-top3-terra",
                               correct_text=P + "long-dev-v1-r1-correct-symspell-terra",
                               invoke_vlm=P + "long-dev-v1-r1-visual-terra")),
    "validation": dict(pilot="ohr_val_multipage_v1", cases=P + "val-multipage-v1-confirmation",
                       diag=P + "val-multipage-v1-diagnosis", base=P + "val-multipage-v1-baseline-answer-v2-terra",
                       runs=dict(retry_retrieval=P + "val-multipage-v1-retry-keep-top3-terra",
                                 correct_text=P + "val-multipage-v1-correct-symspell-terra",
                                 invoke_vlm=P + "val-multipage-v1-visual-terra")),
    "test": dict(pilot="ohr_test_multipage_v1", cases=P + "test-multipage-v1-final-test",
                 diag=P + "test-multipage-v1-diagnosis", base=P + "test-multipage-v1-baseline-answer-v2-terra",
                 runs=dict(retry_retrieval=P + "test-multipage-v1-retry-keep-top3-terra",
                           correct_text=P + "test-multipage-v1-correct-symspell-terra",
                           invoke_vlm=P + "test-multipage-v1-visual-terra")),
}
REPAIRS = ["retry_retrieval", "correct_text", "invoke_vlm"]
REFS = {"no_recovery": (0.236, 0.00171), "faar": (0.266, 0.00320), "gated_image": (0.315, 0.00831),
        "hindsight_best": (0.346, 0.00244)}
LAYOUT = ["excess_whitespace", "bare_numbers", "length_variance", "pipe_heavy", "fraction_heavy"]
GATE = ["abstained", "document_ocr_gap", "low_query_token_coverage"]
QSTART = {"what": ["what"], "how": ["how"], "which": ["which"], "who": ["who", "whose", "whom"],
          "when": ["when"], "where": ["where"],
          "yesno": ["is", "are", "was", "were", "does", "do", "did", "can", "has", "have", "will", "should"]}


def jl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_sample(name):
    cfg = SAMPLES[name]
    cs = pd.read_csv(ANA / cfg["cases"] / "case_scores.csv")
    diag = {d["question_id"]: d for d in jl(DEV / cfg["diag"] / "diagnosis.jsonl")}
    base = {d["question_id"]: d for d in jl(DEV / cfg["base"] / "predictions.jsonl")}
    runs = {r: {d["question_id"]: d for d in jl(DEV / run / "predictions.jsonl")} for r, run in cfg["runs"].items()}
    man = json.load(open(PIL / cfg["pilot"] / "runtime_manifest.json"))
    docs = {d["doc_id"]: d for d in man["documents"]}
    qtext = {q["question_id"]: q["question"] for q in man["questions"]}
    rows = []
    for c in cs.itertuples(index=False):
        c = dict(zip(cs.columns, c))
        q, d, b, doc = c["question_id"], diag[c["question_id"]], base[c["question_id"]], docs[c["doc_id"]]
        assert d["flagged"] == bool(c["flagged"]) and json.dumps(d["gate_reasons"]) == c["gate_reasons"], q
        r = dict(sample=name, question_id=q, doc_id=c["doc_id"], flagged=bool(c["flagged"]),
                 gate_combo="+".join(d["gate_reasons"]) or "none", diagnosis=d["diagnosis"],
                 em_none=float(c["no_recovery:em"]), cost_base=float(b["measured_cost"]))
        for rep in REPAIRS:  # labels and costs; blank/missing/skipped repair keeps the initial answer
            v = c.get(f"diagnostic:{rep}:em")
            p = runs[rep].get(q)
            r[f"em_{rep}"] = r["em_none"] if (v is None or pd.isna(v)) else float(v)
            mc = p.get("measured_cost") if p else None
            r[f"cost_{rep}"] = float(mc) if mc is not None else 0.0  # skipped or failed repair: no recorded cost
            r[f"costmissing_{rep}"] = bool(p is not None and mc is None)
            r[f"skipped_{rep}"] = bool(p is None or p["status"] != "answered")
            r[f"abst_{rep}"] = bool(p and p["status"] == "answered" and p["abstained"])  # runtime output of the repair
        # ---------------- runtime features (prefix f_) ----------------
        f, s = {}, d["signals"]
        f.update(d["features"])
        f["weird_char_ratio"] = s["weird_char_ratio"]
        f["layout_signal_count"] = s["layout_signal_count"]
        for k in LAYOUT:
            f[f"layout_{k}"] = int(s["layout_signals"][k])
        f["initial_abstained"] = int(d["initial_abstained"])
        for g in GATE:
            f[f"gate_{g}"] = int(g in d["gate_reasons"])
        f["n_gate_reasons"] = len(d["gate_reasons"])
        for g in ["semantic", "word_level", "structural", "missing_text"]:
            f[f"diag_{g}"] = int(d["diagnosis"] == g)
        f["n_shown_pages"] = len(s["shown_pages"])
        f["n_shown_pages_ocr_gap"] = len(s["shown_pages_with_ocr_gap"])
        status = {p["page_idx"]: p["ocr_status"] for p in doc["pages"]}
        gaps = sorted(i for i, st in status.items() if st != "ok")
        npages = len(doc["pages"])
        f["doc_page_count"] = npages
        f["doc_log_page_count"] = float(np.log(npages))
        f["doc_n_gap_pages"] = len(gaps)
        f["doc_share_pages_empty_ocr"] = len(gaps) / npages
        f["doc_pdf_kb_per_page"] = doc["pdf"]["bytes"] / 1024 / npages
        f["doc_chunks"] = b["ocr_condition"]["chunks"]
        f["doc_chunks_per_page"] = b["ocr_condition"]["chunks"] / npages
        ev = sorted(b["evidence"], key=lambda e: e["rank"])
        pages = [e["page_idx"] for e in ev]
        f["retrieved_any_page_ocr_gap"] = int(any(status[p] != "ok" for p in pages))
        dist = min((abs(p - g) for p in pages for g in gaps), default=999)
        f["retrieved_adjacent_to_gap_page"] = int(dist <= 1)
        f["retrieved_near_gap_page_le3"] = int(dist <= 3)
        f["n_retrieved_chunks"] = len(ev)
        f["n_distinct_retrieved_pages"] = len(set(pages))
        f["retrieved_page_span"] = max(pages) - min(pages)
        f["top_page_rel_position"] = pages[0] / max(npages - 1, 1)
        f["top_chunk_chars"] = ev[0]["chars"]
        f["mean_chunk_chars"] = float(np.mean([e["chars"] for e in ev]))
        f["total_evidence_chars"] = sum(e["chars"] for e in ev)
        f["top_bm25_score"] = ev[0]["bm25_score"]
        f["top_dense_score"] = ev[0]["dense_score"]
        f["fused_top_minus_last"] = ev[0]["fused_score"] - ev[-1]["fused_score"]
        f["mean_fused_score"] = float(np.mean([e["fused_score"] for e in ev]))
        f["query_retrieval_tokens"] = b["query_retrieval_tokens"]
        f["initial_input_tokens"] = b["usage"]["input_tokens"]
        f["initial_output_tokens"] = b["usage"]["output_tokens"]
        ans = b["answer"] or ""
        f["initial_answer_chars"] = len(ans)
        f["initial_answer_words"] = len(ans.split())
        f["initial_answer_has_digit"] = int(bool(re.search(r"\d", ans)))
        qt = qtext[q]
        words = qt.split()
        f["question_words"] = len(words)
        f["question_chars"] = len(qt)
        f["question_has_digit"] = int(bool(re.search(r"\d", qt)))
        f["question_has_year"] = int(bool(re.search(r"\b(19|20)\d\d\b", qt)))
        f["question_has_money_or_pct"] = int(bool(re.search(r"[$%€£]|percent", qt.lower())))
        f["question_how_many_much"] = int(bool(re.search(r"\bhow (many|much)\b", qt.lower())))
        first = re.sub(r"[^a-z]", "", words[0].lower()) if words else ""
        for k, ws in QSTART.items():
            f[f"q_starts_{k}"] = int(first in ws)
        f["_doc_type"] = c["doc_id"].split("/")[0]
        r.update({f"f_{k}": v for k, v in f.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def add_classes(df):
    text = (df.em_retry_retrieval == 1) | (df.em_correct_text == 1)
    img = df.em_invoke_vlm == 1
    df["cls"] = np.select([text & ~img, img & ~text, text & img], ["text_only", "image_only", "both"], "neither")
    df.loc[~df.flagged, "cls"] = "unflagged"
    return df


def auc(y, x):
    """AUC of x for y==1 (ties count half). nan if one class is absent."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def doc_boot_indices(doc_ids, rng, n=N_BOOT):
    codes, uniq = pd.factorize(np.asarray(doc_ids))
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[i] for i in pick])


def ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [None, None]


def univariate(df, feats, flags, seed):
    """text_only vs image_only; positive class = image_only. Document bootstrap."""
    d = df[df.cls.isin(["text_only", "image_only"])].reset_index(drop=True)
    y = (d.cls == "image_only").to_numpy()
    X = d[feats].to_numpy(float)
    idx = list(doc_boot_indices(d.doc_id, np.random.default_rng(seed)))
    out = {}
    for j, f in enumerate(feats):
        x = X[:, j]
        res = dict(n_image_only=int(y.sum()), n_text_only=int((~y).sum()), auc=auc(y, x),
                   auc_ci=ci([auc(y[i], x[i]) for i in idx]),
                   mean_image_only=float(x[y].mean()), mean_text_only=float(x[~y].mean()))
        if f in flags:
            rd = lambda yy, xx: (xx[yy].mean() - xx[~yy].mean()) if yy.any() and (~yy).any() else np.nan
            res["rate_diff"] = float(rd(y, x))
            res["rate_diff_ci"] = ci([rd(y[i], x[i]) for i in idx])
        out[f] = res
    return out


def policy_stats(df, action):
    """EM and measured cost per question over all rows. action: array of 'none'|repair name (unflagged forced none)."""
    action = np.where(df.flagged.to_numpy(), action, "none")
    em = df.em_none.to_numpy().copy()
    cost = df.cost_base.to_numpy().copy()
    for rep in REPAIRS:
        m = action == rep
        em[m] = df[f"em_{rep}"].to_numpy()[m]
        cost[m] += df[f"cost_{rep}"].to_numpy()[m]
    return em, cost, action


def summarise(df, action, label, extra=None):
    em, cost, action = policy_stats(df, action)
    fl = df.flagged.to_numpy()
    r = dict(policy=label, em=float(em.mean()), cost_per_question=float(cost.mean()),
             share_flagged_to_image=float((action[fl] == "invoke_vlm").mean()) if fl.any() else 0.0,
             share_flagged_to_text=float(np.isin(action[fl], REPAIRS[:2]).mean()) if fl.any() else 0.0)
    r.update(extra or {})
    return r, em


def references(df):
    n = len(df)
    full = lambda a: np.full(n, a, dtype=object)
    out = {}
    out["no_recovery"], _ = summarise(df, full("none"), "no_recovery")
    out["all_retry_retrieval"], _ = summarise(df, full("retry_retrieval"), "all_retry_retrieval")
    out["all_correct_text"], _ = summarise(df, full("correct_text"), "all_correct_text")
    out["gated_image"], _ = summarise(df, full("invoke_vlm"), "gated_image")
    out["faar_diagnosis_route"], _ = summarise(
        df, df.diagnosis.map({"semantic": "retry_retrieval", "word_level": "correct_text",
                              "structural": "invoke_vlm", "missing_text": "invoke_vlm"}).to_numpy(object),
        "faar_diagnosis_route")
    # hindsight best: cheapest action among those with the top EM (none < retry < correct < image)
    ems = np.stack([df.em_none] + [df[f"em_{r}"] for r in REPAIRS], 1)
    order = np.array(["none"] + REPAIRS, dtype=object)
    out["hindsight_best"], _ = summarise(df, order[ems.argmax(1)], "hindsight_best")
    return out


def main():
    res = {"seed": SEED, "n_boot": N_BOOT, "references_from_test_report": REFS}
    data = {s: add_classes(load_sample(s)) for s in SAMPLES}
    # ---- 1. classes
    res["class_counts"] = {}
    for s, df in data.items():
        fl = df[df.flagged]
        cc = fl.cls.value_counts().to_dict()
        res["class_counts"][s] = dict(questions=len(df), documents=int(df.doc_id.nunique()), flagged=len(fl),
                                      **{k: int(cc.get(k, 0)) for k in ["text_only", "image_only", "both", "neither"]},
                                      initial_right_among_flagged=int(fl.em_none.sum()),
                                      skipped_or_failed={r: int(fl[f"skipped_{r}"].sum()) for r in REPAIRS},
                                      cost_field_null={r: int(fl[f"costmissing_{r}"].sum()) for r in REPAIRS})
    assert res["class_counts"]["test"]["text_only"] == 27 and res["class_counts"]["test"]["image_only"] == 71
    assert res["class_counts"]["validation"]["text_only"] == 35 and res["class_counts"]["validation"]["image_only"] == 57
    val_docs = set(data["validation"].doc_id)
    data["test"]["doc_in_validation"] = data["test"].doc_id.isin(val_docs)
    res["doc_overlap"] = dict(test_docs=int(data["test"].doc_id.nunique()),
                              test_docs_in_validation=int(data["test"][data["test"].doc_in_validation].doc_id.nunique()),
                              test_docs_in_practice=int(len(set(data["test"].doc_id) & set(data["practice"].doc_id))),
                              question_id_overlap_test_validation=int(len(set(data["test"].question_id) & set(data["validation"].question_id))))

    # ---- 2. feature matrix (doc_type one-hot from train categories)
    train_all = pd.concat([data["practice"], data["validation"]], ignore_index=True)
    test_all = data["test"].reset_index(drop=True)
    types = sorted(train_all["f__doc_type"].unique())
    for df in (train_all, test_all):
        for t in types:
            df[f"f_doc_type_{t}"] = (df["f__doc_type"] == t).astype(int)
    feats = [c for c in train_all.columns if c.startswith("f_") and c != "f__doc_type"]
    flags = {f for f in feats if set(pd.concat([train_all[f], test_all[f]]).unique()) <= {0, 1}}
    const = [f for f in feats if train_all[train_all.flagged][f].nunique() <= 1]
    res["features"] = dict(all=[f[2:] for f in feats], binary=sorted(f[2:] for f in flags),
                           constant_on_flagged_train=[f[2:] for f in const])
    tr = train_all[train_all.flagged].reset_index(drop=True)
    te = test_all[test_all.flagged].reset_index(drop=True)
    te_new = te[~te.doc_in_validation].reset_index(drop=True)

    # ---- 3. univariate
    uni = {"train_pooled": univariate(tr, feats, flags, SEED + 1), "test": univariate(te, feats, flags, SEED + 2),
           "test_docs_not_in_validation": univariate(te_new, feats, flags, SEED + 3)}
    res["univariate"] = {k: {f[2:]: v for f, v in u.items()} for k, u in uni.items()}

    # ---- 4. routers
    mfeats = [f for f in feats if f not in const]
    Xtr, Xte = tr[mfeats].to_numpy(float), te[mfeats].to_numpy(float)
    groups = tr.doc_id.to_numpy()
    y_img = (tr.cls == "image_only").to_numpy().astype(int)       # image right and both text repairs wrong
    yte_img = (te.cls == "image_only").to_numpy().astype(int)
    text_em_train = {r: float(tr[f"em_{r}"].mean()) for r in REPAIRS[:2]}
    TEXT = max(text_em_train, key=text_em_train.get)
    res["router"] = dict(n_train_flagged=len(tr), n_train_docs=int(tr.doc_id.nunique()), train_positive=int(y_img.sum()),
                         test_positive=int(yte_img.sum()), text_repair_em_on_train_flagged=text_em_train,
                         better_text_repair_on_train=TEXT, n_model_features=len(mfeats))

    def make(kind):
        if kind.startswith("logreg"):
            return make_pipeline(StandardScaler(), LogisticRegression(C=float(kind.split("C")[1]), max_iter=5000))
        if kind == "tree_d3":
            return DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED)
        return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=1.0,
                                          random_state=SEED)

    def oof(kind, y):
        p = np.zeros(len(y))
        for a, b in GroupKFold(n_splits=5).split(Xtr, y, groups):
            p[b] = make(kind).fit(Xtr[a], y[a]).predict_proba(Xtr[b])[:, 1]
        return p

    kinds = ["logreg_C0.01", "logreg_C0.1", "logreg_C1", "tree_d3", "gboost_d2"]
    oofs = {k: oof(k, y_img) for k in kinds}
    cv_auc = {k: auc(y_img, oofs[k]) for k in kinds}
    best_lr = max(kinds[:3], key=cv_auc.get)
    chosen = {"logreg": best_lr, "tree": "tree_d3", "gboost": "gboost_d2"}
    res["router"]["cv_auc_all_candidates"] = cv_auc
    res["router"]["logreg_C_chosen_by_grouped_cv"] = best_lr
    new_mask = (~te.doc_in_validation).to_numpy()
    models, ptest = {}, {}
    res["router"]["binary_models"] = {}
    for name, kind in chosen.items():
        m = make(kind).fit(Xtr, y_img)
        models[name], ptest[name] = m, m.predict_proba(Xte)[:, 1]
        bi = list(doc_boot_indices(tr.doc_id, np.random.default_rng(SEED + 10)))
        ti = list(doc_boot_indices(te.doc_id, np.random.default_rng(SEED + 11)))
        ni = list(doc_boot_indices(te.doc_id[new_mask], np.random.default_rng(SEED + 12)))
        pn, yn = ptest[name][new_mask], yte_img[new_mask]
        res["router"]["binary_models"][name] = dict(
            kind=kind, cv_auc=cv_auc[kind], cv_auc_ci=ci([auc(y_img[i], oofs[kind][i]) for i in bi]),
            test_auc=auc(yte_img, ptest[name]), test_auc_ci=ci([auc(yte_img[i], ptest[name][i]) for i in ti]),
            test_new_docs_auc=auc(yn, pn), test_new_docs_auc_ci=ci([auc(yn[i], pn[i]) for i in ni]),
            test_new_docs_n=int(new_mask.sum()), test_new_docs_positive=int(yn.sum()))
    # does the score add anything beyond the abstention flag, and does it separate text_only from image_only?
    ab_tr, ab_te = (tr.f_initial_abstained == 1).to_numpy(), (te.f_initial_abstained == 1).to_numpy()
    to_tr, to_te = tr.cls.isin(["text_only", "image_only"]).to_numpy(), te.cls.isin(["text_only", "image_only"]).to_numpy()
    for name, kind in chosen.items():
        m = res["router"]["binary_models"][name]
        for key, mtr, mte in [("within_initial_abstained", ab_tr, ab_te), ("text_only_vs_image_only", to_tr, to_te)]:
            bi = list(doc_boot_indices(tr.doc_id[mtr], np.random.default_rng(SEED + 13)))
            ti = list(doc_boot_indices(te.doc_id[mte], np.random.default_rng(SEED + 14)))
            a, b, c, d = y_img[mtr], oofs[kind][mtr], yte_img[mte], ptest[name][mte]
            m[key] = dict(cv_n=int(mtr.sum()), cv_positive=int(a.sum()), cv_auc=auc(a, b), cv_auc_ci=ci([auc(a[i], b[i]) for i in bi]),
                          test_n=int(mte.sum()), test_positive=int(c.sum()), test_auc=auc(c, d), test_auc_ci=ci([auc(c[i], d[i]) for i in ti]))
    res["router"]["initial_abstained_flag_alone"] = dict(cv_auc=auc(y_img, ab_tr), test_auc=auc(yte_img, ab_te),
        train_positives_among_abstained=int(y_img[ab_tr].sum()), train_positives_among_not_abstained=int(y_img[~ab_tr].sum()),
        test_positives_among_abstained=int(yte_img[ab_te].sum()), test_positives_among_not_abstained=int(yte_img[~ab_te].sum()),
        test_abstained=int(ab_te.sum()), test_not_abstained=int((~ab_te).sum()))
    lr = models["logreg"]
    coef = lr[-1].coef_[0]
    res["router"]["logreg_top_coefficients_standardised"] = [
        [mfeats[i][2:], float(coef[i])] for i in np.argsort(-np.abs(coef))[:12]]
    gb = models["gboost"]
    res["router"]["gboost_top_importances"] = [
        [mfeats[i][2:], float(gb.feature_importances_[i])] for i in np.argsort(-gb.feature_importances_)[:12]]
    res["router"]["tree_rules"] = export_text(models["tree"], feature_names=[f[2:] for f in mfeats])

    # policy simulation -------------------------------------------------
    def sim_binary(df_all, df_fl, p, t, fallback):
        act = pd.Series("none", index=df_all.question_id, dtype=object)
        act[df_fl.question_id.to_numpy()] = np.where(p >= t, "invoke_vlm", fallback)
        return act.to_numpy(object)

    def random_mix_em(df_all, share, fallback):
        """Expected EM of routing a random `share` of flagged questions to image, the rest to `fallback`."""
        fb = df_all.em_none if fallback == "none" else df_all[f"em_{fallback}"]
        return np.where(df_all.flagged, share * df_all.em_invoke_vlm + (1 - share) * fb, df_all.em_none)

    def sweep(df_all, df_fl, p_fl, p_train_oof, fallback, seed):
        shares = np.round(np.arange(0, 1.0001, 0.1), 2)
        ths = [np.inf] + [float(np.quantile(p_train_oof, 1 - s)) for s in shares[1:-1]] + [-np.inf]
        # threshold picked on train OOF only: maximise train policy EM (ties -> fewer images)
        tr_em = []
        for t in ths:
            a = np.where(p_train_oof >= t, "invoke_vlm", fallback)
            e = np.where(a == "invoke_vlm", tr.em_invoke_vlm, tr.em_none if fallback == "none" else tr[f"em_{fallback}"])
            tr_em.append(e.mean())
        pick = int(np.argmax(tr_em))
        boot = list(doc_boot_indices(df_all.doc_id, np.random.default_rng(seed), 1000))
        rows = []
        for k, (s, t) in enumerate(zip(shares, ths)):
            r, em = summarise(df_all, sim_binary(df_all, df_fl, p_fl, t, fallback), f"image if p>=t else {fallback}")
            d = em - random_mix_em(df_all, r["share_flagged_to_image"], fallback)
            r.update(train_image_share_target=float(s), threshold=None if np.isinf(t) else t,
                     train_oof_em_flagged=float(tr_em[k]), picked_on_train=bool(k == pick),
                     em_random_same_image_share=float(r["em"] - d.mean()), em_gain_vs_random=float(d.mean()),
                     em_gain_vs_random_ci=ci([d[i].mean() for i in boot]))
            rows.append(r)
        return rows

    test_new_all = test_all[~test_all.doc_in_validation].reset_index(drop=True)
    res["references_recomputed"] = {"test": references(test_all), "test_docs_not_in_validation": references(test_new_all),
                                    "train_pooled": references(train_all)}
    res["policy_sweeps"] = {}
    for name, kind in chosen.items():
        for fb in [TEXT, "none"]:
            res["policy_sweeps"][f"{name}|fallback={fb}|test"] = sweep(test_all, te, ptest[name], oofs[kind], fb, SEED + 20)
            res["policy_sweeps"][f"{name}|fallback={fb}|test_docs_not_in_validation"] = sweep(
                test_new_all, te_new, ptest[name][new_mask], oofs[kind], fb, SEED + 21)
            res["policy_sweeps"][f"{name}|fallback={fb}|train_out_of_fold"] = sweep(
                train_all, tr, oofs[kind], oofs[kind], fb, SEED + 22)

    # three-way: one logistic model per action for P(action gives EM=1); choose argmax p_a - lambda * cost_a
    acts = ["none", TEXT, "invoke_vlm"]
    ycol = {"none": "em_none", TEXT: f"em_{TEXT}", "invoke_vlm": "em_invoke_vlm"}
    inc_cost = {"none": 0.0, TEXT: float(tr[f"cost_{TEXT}"].mean()), "invoke_vlm": float(tr.cost_invoke_vlm.mean())}
    p3_te, p3_oof, three_auc = {}, {}, {}
    for a in acts:
        y = tr[ycol[a]].to_numpy().astype(int)
        p3_oof[a] = oof(best_lr, y)
        p3_te[a] = make(best_lr).fit(Xtr, y).predict_proba(Xte)[:, 1]
        three_auc[a] = dict(cv_auc=auc(y, p3_oof[a]), test_auc=auc(te[ycol[a]].to_numpy().astype(int), p3_te[a]))
    lambdas = [0, 2, 5, 10, 15, 20, 30, 50, 100]
    cvec = np.array([inc_cost[a] for a in acts])
    oof_mat, te_mat = np.stack([p3_oof[a] for a in acts], 1), np.stack([p3_te[a] for a in acts], 1)
    tr_em3 = []
    for lam in lambdas:
        ch = (oof_mat - lam * cvec).argmax(1)
        tr_em3.append(float(np.stack([tr[ycol[a]] for a in acts], 1)[np.arange(len(tr)), ch].mean()))
    rows3 = {"test": [], "test_docs_not_in_validation": []}
    for lam, e in zip(lambdas, tr_em3):
        ch = np.array(acts, dtype=object)[(te_mat - lam * cvec).argmax(1)]
        for key, df_all, mask in [("test", test_all, np.ones(len(te), bool)), ("test_docs_not_in_validation", test_new_all, new_mask)]:
            act = pd.Series("none", index=df_all.question_id, dtype=object)
            act[te.question_id.to_numpy()[mask]] = ch[mask]
            r, _ = summarise(df_all, act.to_numpy(object), "three-way argmax p - lambda*cost",
                             dict(lambda_per_usd=lam, train_oof_em_flagged=e))
            r["share_flagged_no_repair"] = float(1 - r["share_flagged_to_image"] - r["share_flagged_to_text"])
            rows3[key].append(r)
    res["three_way"] = dict(actions=acts, mean_incremental_cost_train=inc_cost, per_action_models=three_auc, sweeps=rows3)

    # fixed rules (stated in advance, not tuned): runtime signals incl. the cheap repair's own abstention
    def rules(df_all, seed=SEED + 30):
        ab = df_all.f_initial_abstained.to_numpy() == 1
        out = []
        boot = list(doc_boot_indices(df_all.doc_id, np.random.default_rng(seed)))
        em_img = np.where(df_all.flagged, df_all.em_invoke_vlm, df_all.em_none)

        def vs_image(em):
            d = np.asarray(em, float) - em_img
            return dict(em_minus_gated_image=float(d.mean()), em_minus_gated_image_ci=ci([d[i].mean() for i in boot]))
        for label, act in [
            ("image if initial abstained else no repair", np.where(ab, "invoke_vlm", "none")),
            (f"image if initial abstained else {TEXT}", np.where(ab, "invoke_vlm", TEXT)),
            (f"{TEXT} if initial abstained else no repair", np.where(ab, TEXT, "none")),
        ]:
            r, em = summarise(df_all, act.astype(object), label)
            out.append({**r, **vs_image(em)})
        # cascade: run TEXT on every flagged question; if it abstains (or was skipped), also run image
        fl = df_all.flagged.to_numpy()
        esc = fl & (df_all[f"abst_{TEXT}"].to_numpy() | df_all[f"skipped_{TEXT}"].to_numpy())
        for label, m_run in [(f"cascade: {TEXT}, then image if it abstains", fl),
                             (f"cascade only if initial abstained: {TEXT}, then image if it abstains", fl & ab)]:
            e2 = esc & m_run
            em = np.where(e2, df_all.em_invoke_vlm, np.where(m_run, df_all[f"em_{TEXT}"], df_all.em_none))
            cost = df_all.cost_base + np.where(m_run, df_all[f"cost_{TEXT}"], 0) + np.where(e2, df_all.cost_invoke_vlm, 0)
            out.append(dict(policy=label, em=float(em.mean()), cost_per_question=float(cost.mean()),
                            share_flagged_to_image=float(e2[fl].mean()), share_flagged_to_text=float(m_run[fl].mean()),
                            **vs_image(em)))
        return out
    res["fixed_rules"] = {"train_pooled": rules(train_all), "test": rules(test_all),
                          "test_docs_not_in_validation": rules(test_new_all)}

    # ---- 5. gate check
    def gate_table(fl):
        rows = []
        for combo, g in list(fl.groupby("gate_combo")) + [("ALL FLAGGED", fl)]:
            init = g.em_none == 1
            reps = np.stack([g[f"em_{r}"] for r in REPAIRS], 1)
            row = dict(gate_reasons=combo, n=len(g), initial_em=float(init.mean()), initial_right=int(init.sum()),
                       fixed_by_any_repair=int((~init & (reps.max(1) == 1)).sum()),
                       broken_by_any_repair=int((init & (reps.min(1) == 0)).sum()),
                       broken_by_all_repairs=int((init & (reps.max(1) == 0)).sum()))
            for j, r in enumerate(REPAIRS):
                row[f"fixed_{r}"] = int((~init & (reps[:, j] == 1)).sum())
                row[f"broken_{r}"] = int((init & (reps[:, j] == 0)).sum())
            rows.append(row)
        return rows
    res["gate_check"] = {"test": gate_table(te), "validation": gate_table(data["validation"][data["validation"].flagged]),
                         "practice": gate_table(data["practice"][data["practice"].flagged])}
    sem = {}
    for s, df in [("test", te), ("validation", data["validation"][data["validation"].flagged]),
                  ("practice", data["practice"][data["practice"].flagged])]:
        g = df[df.diagnosis == "semantic"]
        on = (g.f_n_shown_pages_ocr_gap > 0) | (g.f_retrieved_any_page_ocr_gap == 1)
        doc_gap = g.f_doc_n_gap_pages > 0
        sem[s] = dict(flagged_semantic=len(g), gap_on_retrieved_or_shown_page=int(on.sum()),
                      gap_only_elsewhere_in_document=int((doc_gap & ~on).sum()), no_gap_in_document=int((~doc_gap).sum()),
                      of_elsewhere_retrieved_page_adjacent_to_gap=int((doc_gap & ~on & (g.f_retrieved_adjacent_to_gap_page == 1)).sum()))
    res["semantic_ocr_gap_location"] = sem
    res["retrieved_page_with_ocr_gap_all_flagged"] = {s: int(df[df.flagged].f_retrieved_any_page_ocr_gap.sum()) for s, df in data.items()}

    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_report(res, TEXT)
    print("wrote", OUT / "results.json", OUT / "report.md")


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def write_report(res, TEXT):
    L = ["# Router signal exploration (2026-11-15)\n",
         "Offline. Features are runtime-only; EM columns are used as targets and for scoring. Intervals are 95% "
         f"percentile bootstraps resampling documents (seed {res['seed']}). Practice diagnosis file used: "
         "`results/development/2026-11-01-ohr-long-dev-v1-r1-diagnosis/diagnosis.jsonl` (the gate-choice folder holds "
         "no diagnosis jsonl; this one matches case_scores on flagged, gate reasons and diagnosis for 423/423).\n",
         "## 1. Outcome classes among flagged questions\n"]
    cc = res["class_counts"]
    L.append(table(["sample", "questions", "docs", "flagged", "text_only", "image_only", "both", "neither", "initial right (flagged)"],
                   [[s, c["questions"], c["documents"], c["flagged"], c["text_only"], c["image_only"], c["both"], c["neither"],
                     c["initial_right_among_flagged"]] for s, c in cc.items()]))
    o = res["doc_overlap"]
    L.append(f"Test documents also in validation: {o['test_docs_in_validation']} of {o['test_docs']} "
             f"(in practice: {o['test_docs_in_practice']}; shared question ids: {o['question_id_overlap_test_validation']}).\n")
    L.append("## 2. Features\n")
    L.append(f"{len(res['features']['all'])} runtime features: " + ", ".join(res["features"]["all"]) + ".\n")
    L.append("Constant on flagged training questions (dropped from models): " + (", ".join(res["features"]["constant_on_flagged_train"]) or "none") + ".\n")
    L.append("`retrieved_any_page_ocr_gap` is 0 for every flagged question in all samples: a page without OCR text yields "
             "no chunk, so it cannot be retrieved. Gap pages only reach `shown_pages` (which then forces the missing_text diagnosis).\n")
    L.append("## 3. Univariate: text_only vs image_only\n")
    L.append("AUC > 0.5 means higher values go with image_only. Rate diff = rate in image_only minus rate in text_only (flags only). "
             "Sorted by |AUC-0.5| on practice+validation. No multiple-comparison correction; with this many features a few intervals exclude 0.5 by chance.\n")
    u = res["univariate"]
    order = sorted(u["train_pooled"], key=lambda f: -abs((u["train_pooled"][f]["auc"] or 0.5) - 0.5))
    a = u["train_pooled"][order[0]]; b = u["test"][order[0]]; c = u["test_docs_not_in_validation"][order[0]]
    L.append(f"n: practice+validation {a['n_text_only']} text_only / {a['n_image_only']} image_only; test {b['n_text_only']} / {b['n_image_only']}; "
             f"test docs not in validation {c['n_text_only']} / {c['n_image_only']}.\n")
    def cell(x, k):
        return f"{fmt(x[k])} [{fmt(x[k+'_ci'][0])}, {fmt(x[k+'_ci'][1])}]" if k in x else ""
    rows = []
    for f in order:
        x, y, z = u["train_pooled"][f], u["test"][f], u["test_docs_not_in_validation"][f]
        tr_sig = x["auc_ci"][0] is not None and (x["auc_ci"][0] > 0.5 or x["auc_ci"][1] < 0.5)
        te_sig = y["auc_ci"][0] is not None and (y["auc_ci"][0] > 0.5 or y["auc_ci"][1] < 0.5)
        same = (x["auc"] - 0.5) * (y["auc"] - 0.5) > 0
        rows.append([f, cell(x, "auc"), cell(x, "rate_diff"), cell(y, "auc"), cell(y, "rate_diff"), fmt(z["auc"]),
                     "yes" if (tr_sig and te_sig and same) else ""])
    L.append(table(["feature", "train AUC [CI]", "train rate diff [CI]", "test AUC [CI]", "test rate diff [CI]", "test new-docs AUC", "CI excludes 0.5 on both, same side"], rows))
    r = res["router"]
    L.append("## 4. Routers\n")
    L.append(f"Training set: {r['n_train_flagged']} flagged practice+validation questions on {r['n_train_docs']} documents; "
             f"binary target 'image needed' (image right, both text repairs wrong) has {r['train_positive']} positives; test has {r['test_positive']} of 706. "
             f"Text repair EM on flagged training questions: {json.dumps({k: round(v, 4) for k, v in r['text_repair_em_on_train_flagged'].items()})}; "
             f"fallback text repair = **{TEXT}**. 5-fold document-grouped CV; models refit on all training rows and applied once to test.\n")
    L.append(table(["model", "CV AUC [CI]", "test AUC [CI]", f"test new-docs AUC [CI]"],
                   [[f"{n} ({m['kind']})", cell(m, "cv_auc"), cell(m, "test_auc"), cell(m, "test_new_docs_auc")] for n, m in r["binary_models"].items()]))
    nd = next(iter(r["binary_models"].values()))
    L.append(f"New-docs subset: {nd['test_new_docs_n']} flagged questions, {nd['test_new_docs_positive']} positives. "
             f"All CV candidates: {json.dumps({k: round(v, 3) for k, v in r['cv_auc_all_candidates'].items()})}.\n")
    fa = r["initial_abstained_flag_alone"]
    L.append(f"The abstention flag alone: AUC {fa['cv_auc']:.3f} on training, {fa['test_auc']:.3f} on test. Image-needed questions among abstained / not abstained: "
             f"training {fa['train_positives_among_abstained']} / {fa['train_positives_among_not_abstained']}; "
             f"test {fa['test_positives_among_abstained']} of {fa['test_abstained']} / {fa['test_positives_among_not_abstained']} of {fa['test_not_abstained']}.\n")
    for key, title in [("within_initial_abstained", "Same target, only questions whose initial answer abstained (signal beyond the abstention flag)"),
                       ("text_only_vs_image_only", "Model score separating text_only from image_only (the supervisor's question)")]:
        L.append(title + ":\n")
        L.append(table(["model", "CV n / pos", "CV AUC [CI]", "test n / pos", "test AUC [CI]"],
                       [[n, f"{m[key]['cv_n']} / {m[key]['cv_positive']}", cell(m[key], "cv_auc"), f"{m[key]['test_n']} / {m[key]['test_positive']}", cell(m[key], "test_auc")]
                        for n, m in r["binary_models"].items()]))
    L.append("Largest standardised logistic coefficients: " + ", ".join(f"{k} {v:+.2f}" for k, v in r["logreg_top_coefficients_standardised"]) + ".\n")
    L.append("Largest boosting importances: " + ", ".join(f"{k} {v:.2f}" for k, v in r["gboost_top_importances"]) + ".\n")
    L.append("Depth-3 tree:\n\n```\n" + r["tree_rules"] + "```\n")
    L.append("### References\n")
    L.append("From the test report: " + "; ".join(f"{k} {e} EM at {c} USD" for k, (e, c) in res["references_from_test_report"].items()) + ".\n")
    L.append("Recomputed here from case_scores EM and per-question `measured_cost` (baseline cost for every question plus the cost of the repair used; skipped repairs cost 0). Cost field null (run status execution_failed, counted as 0 USD; the question keeps its scored EM): " + json.dumps({s: c["cost_field_null"] for s, c in cc.items()}) + ".\n")
    for key in ["test", "test_docs_not_in_validation"]:
        L.append(f"**{key}**\n")
        L.append(table(["policy", "EM", "USD/question", "flagged to image"],
                       [[k, fmt(v["em"], 4), fmt(v["cost_per_question"], 5), fmt(v["share_flagged_to_image"])] for k, v in res["references_recomputed"][key].items()]))
    L.append("`faar_diagnosis_route` applies the diagnosis route to every flagged question; the reported FAAR keeps the initial answer for a few skipped repairs, hence tiny differences.\n")
    L.append("### Binary router sweeps\n")
    L.append("Thresholds are quantiles of the training out-of-fold probabilities (target share of flagged questions sent to image). "
             "'random same share' is the expected EM of sending the same share of flagged questions to image at random; the gain and its interval are the router's EM minus that. "
             "'picked' marks the threshold with the best out-of-fold training EM (chosen without test).\n")
    for key, rows in res["policy_sweeps"].items():
        L.append(f"**{key}**\n")
        L.append(table(["train share", "test share to image", "EM", "USD/question", "random same share EM", "gain vs random [CI]", "picked"],
                       [[fmt(x["train_image_share_target"], 1), fmt(x["share_flagged_to_image"]), fmt(x["em"], 4), fmt(x["cost_per_question"], 5),
                         fmt(x["em_random_same_image_share"], 4),
                         f"{x['em_gain_vs_random']:+.4f} [{x['em_gain_vs_random_ci'][0]:+.4f}, {x['em_gain_vs_random_ci'][1]:+.4f}]",
                         "picked" if x["picked_on_train"] else ""] for x in rows]))
    t = res["three_way"]
    L.append("### Three-way choice (no repair / text repair / image)\n")
    L.append(f"One logistic model per action for P(EM=1); choose argmax of p minus lambda x mean incremental cost "
             f"({json.dumps({k: round(v, 5) for k, v in t['mean_incremental_cost_train'].items()})} USD). "
             f"Per-action AUC (CV / test): " + "; ".join(f"{a} {m['cv_auc']:.3f} / {m['test_auc']:.3f}" for a, m in t["per_action_models"].items()) + ".\n")
    for key, rows in t["sweeps"].items():
        L.append(f"**{key}**\n")
        L.append(table(["lambda (per USD)", "EM", "USD/question", "to image", "to text", "no repair", "train OOF EM (flagged)"],
                       [[x["lambda_per_usd"], fmt(x["em"], 4), fmt(x["cost_per_question"], 5), fmt(x["share_flagged_to_image"]),
                         fmt(x["share_flagged_to_text"]), fmt(x["share_flagged_no_repair"]), fmt(x["train_oof_em_flagged"], 4)] for x in rows]))
    L.append("### Fixed rules (not tuned)\n")
    L.append("The cascade uses the text repair's own abstention, which is a runtime signal but costs the text repair first.\n")
    for key, rows in res["fixed_rules"].items():
        L.append(f"**{key}**\n")
        L.append(table(["rule", "EM", "USD/question", "flagged to image", "flagged running text repair", "EM minus gated image [CI]"],
                       [[x["policy"], fmt(x["em"], 4), fmt(x["cost_per_question"], 5), fmt(x["share_flagged_to_image"]), fmt(x["share_flagged_to_text"]),
                         f"{x['em_minus_gated_image']:+.4f} [{x['em_minus_gated_image_ci'][0]:+.4f}, {x['em_minus_gated_image_ci'][1]:+.4f}]"] for x in rows]))
    L.append("## 5. Gate check\n")
    L.append("fixed = initial wrong and the repair right; broken = initial right and the repair wrong. 'any' = at least one of the three repairs.\n")
    for s, rows in res["gate_check"].items():
        L.append(f"**{s}**\n")
        L.append(table(["gate reasons", "n", "initial EM", "initial right", "fixed by any", "broken by any", "broken by all",
                        "fixed retry/correct/image", "broken retry/correct/image"],
                       [[x["gate_reasons"], x["n"], fmt(x["initial_em"]), x["initial_right"], x["fixed_by_any_repair"], x["broken_by_any_repair"],
                         x["broken_by_all_repairs"], "/".join(str(x[f"fixed_{k}"]) for k in REPAIRS),
                         "/".join(str(x[f"broken_{k}"]) for k in REPAIRS)] for x in rows]))
    L.append("Flagged questions diagnosed semantic, by where the OCR gap is:\n")
    L.append(table(["sample", "flagged semantic", "gap on a retrieved/shown page", "gap only elsewhere in document", "of those: a retrieved page is next to a gap page", "no gap in document"],
                   [[s, x["flagged_semantic"], x["gap_on_retrieved_or_shown_page"], x["gap_only_elsewhere_in_document"],
                     x["of_elsewhere_retrieved_page_adjacent_to_gap"], x["no_gap_in_document"]] for s, x in res["semantic_ocr_gap_location"].items()]))
    L.append("The zero is by construction: a gap on a shown page makes the diagnosis missing_text, never semantic.\n")
    L.append("## 6. Verdict\n")
    bm, fa, ps = r["binary_models"], r["initial_abstained_flag_alone"], res["policy_sweeps"]
    pt = lambda key, share: next(x for x in ps[key] if abs(x["train_image_share_target"] - share) < 1e-9)
    g = lambda x: f"{x['em']:.4f} EM at {x['cost_per_question']:.5f} USD (vs random at the same image share {x['em_gain_vs_random']:+.4f} [{x['em_gain_vs_random_ci'][0]:+.4f}, {x['em_gain_vs_random_ci'][1]:+.4f}])"
    casc = {k: v[3] for k, v in res["fixed_rules"].items()}
    rule = {k: v[0] for k, v in res["fixed_rules"].items()}
    gt = {x["gate_reasons"]: x for x in res["gate_check"]["test"]}
    na = [x for k, x in gt.items() if "abstained" not in k and k != "ALL FLAGGED"]
    L += ["Measured:\n",
          f"- A signal exists but it is modest. For 'image needed' the document-grouped CV AUC is {bm['logreg']['cv_auc']:.3f} (logistic) and {bm['gboost']['cv_auc']:.3f} (boosting); "
          f"test AUC is {bm['logreg']['test_auc']:.3f} and {bm['gboost']['test_auc']:.3f}, but 252 of 327 test documents are also validation documents, and on the 75 new documents "
          f"({bm['logreg']['test_new_docs_n']} flagged, {bm['logreg']['test_new_docs_positive']} positives) the intervals are too wide to say anything "
          f"({bm['logreg']['test_new_docs_auc']:.3f} and {bm['gboost']['test_new_docs_auc']:.3f}). The CV figure is the document-independent one.",
          f"- Separating text_only from image_only: model score CV AUC {bm['logreg']['text_only_vs_image_only']['cv_auc']:.3f} / {bm['gboost']['text_only_vs_image_only']['cv_auc']:.3f}; the best single features sit at 0.67-0.69 "
          "(score gap between the first and last retrieved chunk, top fused score, top BM25 score: higher goes with image_only; a document with OCR gaps goes with text_only). No layout, weird-character, question-form or doc-type feature holds on both samples.",
          f"- Much of the signal is the abstention flag: on test {fa['test_positives_among_abstained']} of {fa['test_positives_among_abstained'] + fa['test_positives_among_not_abstained']} image-needed questions had an abstained initial answer. "
          f"Within abstained questions the models still rank above chance (CV {bm['logreg']['within_initial_abstained']['cv_auc']:.3f} / {bm['gboost']['within_initial_abstained']['cv_auc']:.3f}).",
          f"- Learned router on test, sending the top ~40% (by training quantile) to image and the rest to retry_retrieval: {g(pt('logreg|fallback=retry_retrieval|test', 0.4))}. "
          f"Out-of-fold on training at the same share: {g(pt('logreg|fallback=retry_retrieval|train_out_of_fold', 0.4))}. "
          "No operating point comes near hindsight best (0.346 at 0.00244).",
          f"- Untuned rule 'image only if the initial answer abstained, otherwise no repair': {rule['test']['em']:.4f} EM at {rule['test']['cost_per_question']:.5f} USD on test, i.e. gated-image EM at about three quarters of its cost.",
          f"- Untuned cascade 'retry_retrieval, then image if the retry abstains': {casc['test']['em']:.4f} EM at {casc['test']['cost_per_question']:.5f} USD on test, "
          f"{casc['test']['em_minus_gated_image']:+.4f} [{casc['test']['em_minus_gated_image_ci'][0]:+.4f}, {casc['test']['em_minus_gated_image_ci'][1]:+.4f}] EM against gated image; "
          f"on practice+validation {casc['train_pooled']['em_minus_gated_image']:+.4f} [{casc['train_pooled']['em_minus_gated_image_ci'][0]:+.4f}, {casc['train_pooled']['em_minus_gated_image_ci'][1]:+.4f}]; "
          f"on the 75 new test documents {casc['test_docs_not_in_validation']['em_minus_gated_image']:+.4f} [{casc['test_docs_not_in_validation']['em_minus_gated_image_ci'][0]:+.4f}, {casc['test_docs_not_in_validation']['em_minus_gated_image_ci'][1]:+.4f}].",
          f"- Gate: the {sum(x['n'] for x in na)} test questions flagged without an abstention had {sum(x['initial_right'] for x in na)} right initially; a repair fixed {sum(x['fixed_by_any_repair'] for x in na)} and at least one repair broke {sum(x['broken_by_any_repair'] for x in na)}. "
          f"For document_ocr_gap alone: n={gt['document_ocr_gap']['n']}, initial EM {gt['document_ocr_gap']['initial_em']:.3f}, fixed {gt['document_ocr_gap']['fixed_by_any_repair']}, broken {gt['document_ocr_gap']['broken_by_any_repair']}.\n",
          "Interpretation:\n",
          "- A learned router is a cost saver, not an accuracy gain: it can roughly hold gated-image EM for about two thirds of the cost, and most of that comes from one flag (did the first answer abstain). The features on file do not identify the text_only cases well enough to approach the hindsight policy.",
          "- The two cheap rules do as well as or better than the learned models. The cascade is the only policy here that beats gated image on EM, because the text repair's own abstention is a stronger signal than anything available before a repair is run.",
          "- The document_ocr_gap and low-coverage gate reasons flag questions that repairs almost never fix; restricting the gate to abstentions would cut cost with no measurable EM loss on these samples.",
          "- All of this is post hoc on samples already used for the study. The rules were written down before their test numbers were computed, but they were not pre-registered, and the test sample shares documents with validation. A fresh document-disjoint sample is needed before any of it is claimed.\n"]
    (OUT / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
