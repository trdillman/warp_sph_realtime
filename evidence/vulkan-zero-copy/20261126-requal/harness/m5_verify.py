"""M5 host-side verdict: applies the pre-registered M5_PROTOCOL.md criteria to runs/<lane>/.

Usage: python m5_verify.py. Writes m5-verdict.json next to runs/, PNGs of every parity image,
and the benchmark table to reports/benchmarks/vulkan-zero-copy-m5-<evidence dir>.{json,md}.
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[3]
RUNS = OUT / 'runs'
BENCH = ROOT / 'reports' / 'benchmarks'
OWNED = ('VulkanInteropTransport', 'ReadbackTransport', 'TransportSession', 'RealtimeWCSPHSolver', 'PointRenderer')
MIB = 1 << 20
GATE_P95_MS = 16.67  # owner-revised 4M gate (docs/VULKAN_ZERO_COPY_PLAN.md M5, 2026-11-26), one 60 Hz frame
INFO_P95_MS, INFO_MAX_COUNT = 2.0, 1_000_000  # informational target at <= 1M particles, reported only


def load(lane, name):
    path = RUNS / lane / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def png(path, rgba):
    h, w, _ = rgba.shape
    raw = b''.join(b'\x00' + rgba[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def within(a, b, tol):
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) <= tol).mean())


def images():
    sets = {lane: dict(np.load(RUNS / lane / 'parity-images.npz')) for lane in ('vulkan', 'opengl')
            if (RUNS / lane / 'parity-images.npz').exists()}
    for lane, imgs in sets.items():
        folder = OUT / 'images' / lane
        folder.mkdir(parents=True, exist_ok=True)
        for name, rgba in imgs.items():
            png(folder / f'{name}.png', rgba[::-1])  # GPU rows are bottom-up
    rows = []
    vk, gl = sets.get('vulkan', {}), sets.get('opengl', {})
    for name in sorted(vk):
        if name.startswith('interop_'):
            ref = 'readback_' + name[len('interop_'):]
            frac = within(vk[name], vk[ref], 1)
            rows.append({'pair': f'vulkan:{name} vs vulkan:{ref}', 'tolerance': '1/255', 'fraction': frac,
                         'threshold': 0.9999, 'pass': frac >= 0.9999})
    for name in sorted(gl):
        if name in vk:
            frac = within(vk[name], gl[name], 2)
            rows.append({'pair': f'vulkan:{name} vs opengl:{name}', 'tolerance': '2/255', 'fraction': frac,
                         'threshold': 0.995, 'pass': frac >= 0.995})
    nonblank = {f'{lane}:{n}': int((im[..., :3].max(axis=2) > 0).sum()) for lane, s in sets.items()
                for n, im in s.items()}
    return rows, nonblank


def envelope(cycles):
    if not cycles:
        return {'pass': False, 'reason': 'cycles.json missing'}
    rows = [r for r in cycles if r['kind'] in ('start_stop', 'resize', 'disable', 'file_load')]
    failures = []
    for r in rows:
        live = {k: v for k, v in r['owned'].items() if v}
        if r['kind'] != 'resize' and live:
            failures.append((r['kind'], r['cycle'], 'live objects', live))
        if r['quarantine']:
            failures.append((r['kind'], r['cycle'], 'quarantine', r['quarantine']))
        if r['last_close'] != 'completed' or r['restart_required'] or r['interop_unavailable']:
            failures.append((r['kind'], r['cycle'], 'close/state', r['last_close'], r['restart_required'],
                             r['interop_unavailable']))
    by_kind = {}
    for kind in ('start_stop', 'disable', 'file_load'):
        ks = [r for r in rows if r['kind'] == kind]
        if len(ks) < 50:
            failures.append((kind, 'cycles', len(ks)))
            continue
        warm = ks[:3]
        base_vram = max(r['vram_used'] for r in warm)
        spread = max(r['vram_used'] for r in warm) - min(r['vram_used'] for r in warm)
        allowance = max(64 * MIB, 3 * spread)
        handles = np.asarray([r['handles'] for r in ks[9:]], dtype=np.float64)
        slope = float(np.polyfit(np.arange(len(handles)), handles, 1)[0])
        base_handles = max(r['handles'] for r in warm)
        entry = {'vram_warm_mib': base_vram / MIB, 'vram_final_mib': ks[-1]['vram_used'] / MIB,
                 'vram_allowance_mib': allowance / MIB, 'handle_slope_per_cycle': slope,
                 'handles_warm': base_handles, 'handles_final': ks[-1]['handles']}
        entry['pass'] = (ks[-1]['vram_used'] <= base_vram + allowance and slope < 0.5
                         and ks[-1]['handles'] <= base_handles + 32)
        by_kind[kind] = entry
        if not entry['pass']:
            failures.append((kind, 'envelope', entry))
    resize = [r for r in rows if r['kind'] == 'resize']
    if len(resize) < 50 or any(r.get('old_close') != 'completed' for r in resize):
        failures.append(('resize', 'cycles/close', len(resize)))
    if resize and max(r['owned']['VulkanInteropTransport'] for r in resize) > 1:
        failures.append(('resize', 'more than one live interop transport'))
    final = next((r for r in cycles if r['kind'] == 'final'), None)
    if final is None or any(final['owned'].values()) or final['quarantine']:
        failures.append(('final', final and final['owned'], final and final['quarantine']))
    return {'pass': not failures, 'failures': failures, 'by_kind': by_kind, 'cycles': len(rows)}


def bench_table(rows):
    lines = ['| lane | kind | count | status | samples | drops | completed p50/p95/max ms | publish→draw p95 ms |'
             ' pack GPU p95 ms | CPU submit p95 ms | frame p50 ms |', '|' + '---|' * 11]
    for r in rows:
        def p(key, field='p95'):
            v = r.get(key)
            return f"{v[field]:.3f}" if isinstance(v, dict) else ('n/a' if v is None else 'unmeasured')
        comp = r.get('completed_latency_ms')
        comp_s = (f"{comp['p50']:.3f}/{comp['p95']:.3f}/{comp['max']:.3f}" if isinstance(comp, dict)
                  else 'unmeasured')
        lines.append(f"| {r['lane']} | {r['kind']} | {r['count']} | {r['status']} | {r['samples']} | {r['drops']} | "
                     f"{comp_s} | {p('publish_to_draw_ms')} | {p('pack_gpu_ms')} | {p('cpu_submit_ms')} | "
                     f"{p('frame_time_ms', 'p50')} |")
    return '\n'.join(lines)


def main():
    verdict = {'lanes': {}}
    for lane in ('vulkan', 'opengl'):
        verdict['lanes'][lane] = {'supervisor': load(lane, 'supervisor.json'), 'probe': load(lane, 'probe.json'),
                                  'identity': load(lane, 'identity.json')}
    parity = {lane: load(lane, 'parity.json') for lane in ('vulkan', 'opengl')}
    verdict['parity'] = {lane: {'rows': len(p or []), 'pass': bool(p) and all(
        r['exact'] and r['padding_zero'] and r['close'] == 'completed' for r in p)} for lane, p in parity.items()}
    verdict['parity']['vulkan']['interop_rows'] = sum(r['kind'] == 'interop' for r in parity['vulkan'] or [])
    img_rows, nonblank = images()
    verdict['images'] = {'pairs': img_rows, 'nonblank_pixels': nonblank,
                         'pass': bool(img_rows) and all(r['pass'] for r in img_rows)
                         and all(v > 0 for v in nonblank.values())}
    verdict['cycles'] = envelope(load('vulkan', 'cycles.json'))
    faults = load('vulkan', 'faults.json')
    verdict['faults'] = {'pass': bool(faults and faults.get('all_pass')),
                         'failed_cases': [c['case'] for c in (faults or {}).get('cases', []) if not c['pass']],
                         'checks': (faults or {}).get('checks')}
    fatal = load('vulkan', 'fatal.json')
    verdict['fatal_mocked'] = {'pass': bool(fatal and fatal['pass']), 'row': fatal}
    bench = [{**r, 'lane': lane} for lane in ('vulkan', 'opengl') for r in (load(lane, 'bench.json') or [])]
    target = next((r for r in bench if r['lane'] == 'vulkan' and r['kind'] == 'interop' and r['count'] == 4194304),
                  None)
    comp = target and target.get('completed_latency_ms')
    verdict['latency_gate'] = {
        'criterion': f'interop completed transport latency p95 <= {GATE_P95_MS} ms at 4194304, zero drops, COMPLETE samples',
        'p95_ms': comp['p95'] if isinstance(comp, dict) else None,
        'drops': target and target['drops'], 'status': target and target['status'],
        'missing_completions': target and target.get('completed_missing'),
        'uploads': target and target.get('uploads'), 'timer_uploads': target and target.get('timer_uploads'),
        'pass': bool(isinstance(comp, dict) and comp['p95'] <= GATE_P95_MS and target['drops'] == 0
                     and target['status'] == 'COMPLETE' and target.get('completed_missing') == 0)}
    verdict['informational_2ms'] = {
        'criterion': f'interop completed p95 <= {INFO_P95_MS} ms at <= {INFO_MAX_COUNT} particles (not a gate)',
        'rows': [{'count': r['count'], 'p95_ms': r['completed_latency_ms']['p95'],
                  'met': r['completed_latency_ms']['p95'] <= INFO_P95_MS} for r in bench
                 if r['lane'] == 'vulkan' and r['kind'] == 'interop' and r['count'] <= INFO_MAX_COUNT
                 and isinstance(r.get('completed_latency_ms'), dict)]}
    gates = ('parity', 'images', 'cycles', 'faults', 'fatal_mocked', 'latency_gate')
    verdict['gates'] = {g: (all(v['pass'] for v in verdict[g].values()) if g == 'parity' else verdict[g]['pass'])
                        for g in gates}
    verdict['m5_pass'] = all(verdict['gates'].values())
    (OUT / 'm5-verdict.json').write_text(json.dumps(verdict, indent=1, default=str), encoding='utf-8')
    BENCH.mkdir(parents=True, exist_ok=True)
    stem = BENCH / f'vulkan-zero-copy-m5-{OUT.name}'
    stem.with_suffix('.json').write_text(json.dumps(bench, indent=1), encoding='utf-8')
    stem.with_suffix('.md').write_text(
        f'# Vulkan zero-copy M5 benchmarks ({OUT.name})\n\nSource: evidence/vulkan-zero-copy/{OUT.name}/runs/.'
        ' Readback is a CPU copy, never zero-copy. Upload GPU time and Readback completion latency are'
        ' unmeasured (no instrumentation); see M5_PROTOCOL.md.\n\n' + bench_table(bench) + '\n', encoding='utf-8')
    print(json.dumps(verdict['gates'], indent=1), 'M5 PASS' if verdict['m5_pass'] else 'M5 FAIL')
    return 0 if verdict['m5_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
