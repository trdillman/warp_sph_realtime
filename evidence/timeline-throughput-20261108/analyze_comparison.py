"""Validate the bounded same-work comparison; no hardware certification claim."""
import copy
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
BASE_SHA = '3db5a2a1013064e26e5df32610cdad15bda01aa036ab75027c832ce6e13af44b'
CAND_SHA = '785c5248c7ec5f2cc194633c1c97a9ebf330577b6f3c3fadcdedb192b75c4717'
STATE_KEYS = ('steps', 'time', 'frame', 'count', 'positions', 'ids', 'dt', 'substeps', 'live_points')

def verify(baseline, candidate):
    assert baseline['config']['package_sha256'] == BASE_SHA, 'baseline artifact mismatch'
    assert candidate['config']['package_sha256'] == CAND_SHA, 'candidate artifact mismatch'
    for arm in (baseline, candidate):
        assert arm['complete'] and arm['error'] is None, 'incomplete or failed run'
        assert [r['trial'] for r in arm['rows']] == list(range(-1, 16)), 'missing/duplicate trials'
        assert all(r['wall_s'] > 0 for r in arm['rows']), 'invalid wall time'
    expected = baseline['rows'][0]
    assert expected['steps'] == 13 and expected['count'] == 100000 and expected['frame'] == 2
    for row in baseline['rows'] + candidate['rows']:
        assert all(row[k] == expected[k] for k in STATE_KEYS), 'workload/final-state mismatch'
    a = [r['wall_s'] for r in baseline['rows'] if not r['warmup']]
    b = [r['wall_s'] for r in candidate['rows'] if not r['warmup']]
    assert statistics.median(b) < statistics.median(a), 'no observed improvement'
    return {
        'scope': 'bounded reset-trial development comparison, not hardware qualification',
        'trials_per_arm': len(a), 'baseline_median_s': statistics.median(a),
        'candidate_median_s': statistics.median(b),
        'ratio_of_medians': statistics.median(a) / statistics.median(b),
        'baseline_range_s': [min(a), max(a)], 'candidate_range_s': [min(b), max(b)],
        'baseline_adjacent_AA_ratios': [a[i] / a[i+1] for i in range(0, 16, 2)],
        'same_work_and_exact_final_positions_ids': True,
    }

if __name__ == '__main__':
    a = json.loads((ROOT/'baseline-trials.json').read_text())
    b = json.loads((ROOT/'candidate-trials.json').read_text())
    result = verify(a, b)
    controls = {}
    for name, mutate in {
        'wrong_positions': lambda d: d['rows'][3].update(positions='corrupt'),
        'changed_substeps': lambda d: d['rows'][3].update(substeps=1),
        'missing_trial': lambda d: d['rows'].pop(),
        'wrong_artifact': lambda d: d['config'].update(package_sha256='stale'),
        'failed_attempt': lambda d: d.update(error='failed'),
    }.items():
        changed = copy.deepcopy(b)
        mutate(changed)
        try:
            verify(a, changed)
        except AssertionError as exc:
            controls[name] = str(exc)
        else:
            raise AssertionError(f'verifier accepted {name}')
    result['receipt_validation_negative_controls'] = controls
    print(json.dumps(result, indent=2))
