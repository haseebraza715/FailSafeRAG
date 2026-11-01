"""Terminal baseline failure list.

B1. A terminal failed answer blocks diagnosis or repair preparation.
B2. An unfinished prediction or another validity blocker is accepted.
B3. Clean baseline preparation changes when only validity metadata changes.
"""

import json

import diagnose_run as dr
import pytest
import run_text_repair as rtr
import run_visual_repair as rvr
from test_text_repair import DOC_A, fake_replay, hit
from test_visual_repair import PILOT, Project

from faar.diagnosis import GateRule, gate
from faar.visual_repair import RepairError


def change(path, **fields):
    data = json.loads(path.read_text())
    data.update(fields)
    path.write_text(json.dumps(data))


def test_failed_terminal_question_flows_through_all_preparations(tmp_path, monkeypatch):
    project = Project(tmp_path)
    replay = fake_replay(project, [hit('d/a-p1-c0', 1, 'Teh agreemnet was sigend.'), *DOC_A[1:]])
    monkeypatch.setattr(dr, 'replay_retrieval', lambda *args: replay)
    monkeypatch.setattr(rtr, 'replay_retrieval', lambda *args: replay)
    project.prepare()
    for repair, method in [('retry_retrieval', 'exclude-seen-v1'), ('correct_text', 'symspell-en-82765')]:
        rtr.prepare(root=tmp_path, repair=repair, method=method, pilot_id=PILOT,
                    baseline_run_dir=project.baseline, run_dir=project.run_dir(repair), run_id=repair,
                    resend_rule='none', provider_config=project.provider_config)
    before = {(name, file): (project.run_dir(name) / file).read_bytes()
              for name in ('repair', 'retry_retrieval', 'correct_text') for file in ('requests.jsonl', 'run_config.json')}
    summary = project.baseline / 'run_summary.json'
    change(summary, valid_baseline=False, valid_baseline_blockers=['1 question(s) are execution_failed'])
    predictions = project.baseline / 'predictions.jsonl'
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]['status'] = 'execution_failed'
    predictions.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    diagnosed = dr.diagnose_run(root=tmp_path, pilot_id=PILOT, baseline_run_dir=project.baseline,
                                out_dir=project.run_dir('diagnosis'), rule=GateRule())
    assert diagnosed[0]['initial_status'] == 'execution_failed'
    assert diagnosed[0]['flagged'] and 'document_ocr_gap' in diagnosed[0]['gate_reasons']
    assert project.prepare()[0]['action'] == 'send'
    for repair, method in [('retry_retrieval', 'exclude-seen-v1'), ('correct_text', 'symspell-en-82765')]:
        records, _ = rtr.prepare(root=tmp_path, repair=repair, method=method, pilot_id=PILOT,
                                 baseline_run_dir=project.baseline, run_dir=project.run_dir(repair), run_id=repair,
                                 resend_rule='none', provider_config=project.provider_config)
        assert records[0]['action'] == 'send'
    assert all((project.run_dir(name) / file).read_bytes() == data for (name, file), data in before.items())


@pytest.mark.parametrize('field,value', [('status', 'pending'), ('unserved', True), ('awaiting_reconciliation', True)])
def test_unfinished_predictions_are_refused(tmp_path, field, value):
    project = Project(tmp_path)
    path = project.baseline / 'predictions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(RepairError):
        rvr.load_baseline(tmp_path, PILOT, project.baseline)


def test_other_blocker_is_refused(tmp_path):
    project = Project(tmp_path)
    change(project.baseline / 'run_summary.json', valid_baseline=False, valid_baseline_blockers=['returned model mismatch'])
    with pytest.raises(RepairError):
        rvr.load_baseline(tmp_path, PILOT, project.baseline)


@pytest.mark.parametrize('field,value', [('mode', 'mock'), ('run_kind', 'engineering_check'), ('canonical', False), ('run_state', 'incomplete')])
def test_noncanonical_or_incomplete_run_is_refused(tmp_path, field, value):
    project = Project(tmp_path)
    config_path = project.baseline / 'run_config.json'
    config = json.loads(config_path.read_text())
    summary_path = project.baseline / 'run_summary.json'
    if field == 'mode':
        config[field] = value
    elif field == 'run_kind':
        config['identity'][field] = value
    elif field == 'canonical':
        config['identity']['pilot_manifest'][field] = value
    else:
        change(summary_path, run_state=value)
    config_path.write_text(json.dumps(config))
    with pytest.raises(RepairError):
        rvr.load_baseline(tmp_path, PILOT, project.baseline)


def test_failure_alone_does_not_flag_but_other_signals_can():
    args = dict(initial_status='execution_failed', abstained=False, features={'query_token_coverage': 0.1})
    assert gate(**args, pages=[{'ocr_status': 'ok'}], rule=GateRule()) == (False, [])
    assert gate(**args, pages=[{'ocr_status': 'missing'}], rule=GateRule())[1] == ['document_ocr_gap']
    assert gate(**args, pages=[{'ocr_status': 'ok'}], rule=GateRule('query_token_coverage', 0.5))[1] == ['low_query_token_coverage']
