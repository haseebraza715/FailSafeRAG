# Demo corpus (`demo_corpus/`)

A tiny synthetic corpus (three one-page documents, ~40 words each) committed to
the repository so the prototype pipeline demo runs offline and gives the same output
every time. It needs no API key, no model download and no GPU. The FAAR study did not use this demo.

## Layout

```
demo_corpus/
├── data/phase0/sample_manifest.csv            # example_id, doc_name, question, correct_answer, page_no
└── artifacts/phase0/ocr_text/
    ├── clean_threshold.txt                    # clean OCR page: "Zephyr compliance requires a threshold of 42%…"
    ├── noisy_threshold.txt                    # same facts, OCR-corrupted: "Z3phyr c0mpliance requires a thr3shold…"
    └── clean_accesslog.txt                    # clean yes/no page
```

This is the same layout the phase-0 pipeline expects for the real benchmark
(`data/phase0/sample_manifest.csv` + `artifacts/phase0/ocr_text/<id>.txt`),
so any `--project-root` pointing at this directory works with the real code.

## The story

The two threshold pages carry the same fact (`42%`). The corrupted page's
word-level noise (`Z3phyr`, `c0mpliance`, `thr3shold`) makes the evidence
untrustworthy: the quality gate flags `word_noise_alert`, the failure is
diagnosed as `word_level`, and recovery routes to OCR-text correction. Because
the offline profile disables ByT5, the correction is a *guarded skip* and the
pipeline answers from the flagged evidence with a measured confidence
(`support_score` 9 on the noisy page vs 10 on the clean page). No visual model
is invoked on any example, so no image model is called.

## Running it

```bash
python scripts/demo/demo_run.py                # milestone trace over all three examples
```

The `faar-demo run-example` CLI accepts the same `--project-root` pointing at
this directory (it enables word-level ByT5 correction by default; on macOS
builds where faiss and torch OpenMP runtimes conflict, that path can crash.
Use the demo script, or `run-benchmark --no-enable-byt5`, instead).
