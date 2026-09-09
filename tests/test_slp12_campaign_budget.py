"""Pure spending arithmetic for a second campaign on a shared RunPod account."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/slp12_runpod.py'
spec = importlib.util.spec_from_file_location('slp12_budget_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_second_run_accounts_for_prior_spend_and_storage():
    result = module.allocation_budget(32, .99, 50, 30, 9.167421410866615, 42)
    assert result['campaign_total_with_reserve_usd'] < 50
    assert result['account_required_usd'] > 32 * .99 + 5
    assert result['prior_spend_usd'] == 9.167421410866615


def test_other_jobs_consume_credit_but_not_this_campaign_authorization():
    alone = module.allocation_budget(8, .99, 50, 30, 9.17, 40)
    shared = module.allocation_budget(8, .99, 50, 30, 9.17, 40, 1.59)
    assert alone['campaign_total_with_reserve_usd'] == shared['campaign_total_with_reserve_usd']
    assert shared['account_required_usd'] - alone['account_required_usd'] == pytest.approx(8 * 1.59)
    with pytest.raises(RuntimeError, match='shared credit'):
        module.allocation_budget(8, .99, 50, 30, 9.17, 21.69, 1.59)


def test_credit_cannot_override_the_original_campaign_ceiling():
    with pytest.raises(RuntimeError, match='campaign cap'):
        module.allocation_budget(40, .99, 50, 30, 9.17, 1000)


def test_explicit_other_job_horizon_preserves_full_slp_allocation():
    result = module.allocation_budget(32, .99, 50, 30, 10.18, 70, 1.59, 12)
    assert result['account_required_usd'] == pytest.approx(32 * (.99 + 10 / 720) + 12 * 1.59 + 5)
    assert result['campaign_total_with_reserve_usd'] < 50
    with pytest.raises(ValueError):
        module.allocation_budget(32, .99, 50, 30, 10.18, 70, 1.59, float('nan'))


def test_low_credit_guard_deletes_only_recorded_pod_and_preserves_volume(tmp_path, monkeypatch):
    import json
    record = tmp_path / 'campaign.json'
    record.write_text(json.dumps({'pod_id': 'slp-only', 'volume_id': 'persist',
                                 'terminate_at': 1000, 'credit_stop_usd': 5}))
    monkeypatch.setattr(module, 'RECORD', record)
    monkeypatch.setattr(module, 'STATE', tmp_path)
    monkeypatch.setattr(module.time, 'time', lambda: 100)
    calls = []
    def fake_run(*args):
        calls.append(args)
        return json.dumps({'clientBalance': 4.99}) if args == ('user',) else '{}'
    monkeypatch.setattr(module, 'run', fake_run)
    module.guard()
    assert calls == [('user',), ('pod', 'delete', 'slp-only')]
    saved = json.loads(record.read_text())
    assert saved['termination_reason'] == 'low_account_credit'
    assert saved['volume_id'] == 'persist'


@pytest.mark.parametrize('hours', [0, -1, float('inf'), float('nan')])
def test_invalid_runtime_cannot_disable_spending_limit(hours):
    with pytest.raises(ValueError):
        module.allocation_budget(hours, .99, 50, 30, 9.17, 100)


def test_checkpoint_retention_requires_verified_snapshots_and_preserves_default(tmp_path):
    import hashlib
    import json
    import sys
    sys.path.insert(0, str(path.parent))
    try:
        from slp12_collect import prune_checkpoints
    finally:
        sys.path.pop(0)
    for n in (1000, 2000, 3000):
        p = tmp_path / f'checkpoint-{n:07d}'
        p.mkdir()
        (p / 'model').write_bytes(str(n).encode())
        (p / 'manifest.json').write_text(json.dumps({'model': hashlib.sha256(str(n).encode()).hexdigest()}))
    prune_checkpoints(tmp_path, 0)
    assert len(list(tmp_path.iterdir())) == 3
    latest = tmp_path / 'checkpoint-0003000/model'
    latest.write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete'):
        prune_checkpoints(tmp_path, 2)
    assert len(list(tmp_path.iterdir())) == 3
    latest.write_bytes(b'3000')
    prune_checkpoints(tmp_path, 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint-0002000', 'checkpoint-0003000']
