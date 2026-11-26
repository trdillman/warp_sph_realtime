"""M5: validate/build the frozen extension once, then install it into one fresh isolated profile per lane."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'warp_sph_realtime'
SDK = ROOT / 'vendor/blender-vulkan-interop/src/blender_vulkan_interop'
# Byte-identical 5.2.0 fbe6228777e7 build with no bundled Warp (the Program Files copy ships Warp 1.18.0 in its own
# site-packages, which the product rightly refuses to replace). Launches still route through blender-gate.
EXE = r'C:\Users\Tyler\Downloads\blender-5.2.0-windows-x64\blender-5.2.0-windows-x64\blender.exe'
PROFILE = SRC / 'render/profiles/blender-5.2.0-fbe6228777e7-windows-amd64-cp313-v1.json'
KEYS = ('RESOURCES', 'CONFIG', 'SCRIPTS', 'DATAFILES', 'EXTENSIONS')


def profile_env(root: Path) -> dict:
    env = dict(os.environ)
    env.update(BGATE_CLASS='cpu', BGATE_MAX_SECONDS='300', BGATE_OFFLOAD='off', BGATE_LABEL='m5-requal-build', BGATE_BLENDER=EXE)
    for key in KEYS:
        target = root / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        env['BLENDER_USER_' + key] = str(target)
    return env


def run(name, args, env, log_dir):
    command = [EXE, '--factory-startup', '--offline-mode', '--command', 'extension', *args]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    (log_dir / f'{name}.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    row = {'name': name, 'command': command, 'exit_code': result.returncode}
    assert result.returncode == 0, row
    return row


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha(member: str) -> str | None:
    """SHA-256 of the tree file a zip member was built from (the bundled SDK is LF-normalized .py/.json)."""
    sdk = '_vendor/blender_vulkan_interop/'
    if member == '_vendor/SDK_DISTRIBUTION.md':
        path = SDK.parents[1] / 'DISTRIBUTION.md'
    elif member.startswith(sdk):
        path = SDK / member[len(sdk):]
        if path.suffix in ('.py', '.json') and path.is_file():
            return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    else:
        path = SRC / member
    return sha(path) if path.is_file() else None

def main():
    identity = json.loads(PROFILE.read_text())['identity']
    exe = Path(EXE)
    assert sha(exe) == identity['exe_sha256'], 'blender.exe is not the qualified exact-build profile'
    assert sha(exe.with_suffix('.pdb')) == identity['pdb_sha256'], 'blender.pdb is not the qualified profile'
    artifact = OUT / 'artifact'
    profiles = OUT / 'profiles'
    for stale in (artifact, profiles):
        assert not stale.exists(), f'{stale} exists: build runs once per freeze'
    artifact.mkdir()
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    # The candidate is the product build: scripts/build_extension.py (validate + build + bundled pinned SDK).
    build_env = profile_env(OUT / 'build-profile')
    command = [sys.executable, str(ROOT / 'scripts/build_extension.py'), '--blender', EXE]
    result = subprocess.run(command, capture_output=True, text=True, env=build_env, timeout=900, cwd=ROOT)
    (logs / 'build_extension.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    rows = [{'name': 'build_extension', 'command': command, 'exit_code': result.returncode}]
    assert result.returncode == 0, rows
    built = list((ROOT / 'dist').glob('warp_sph_realtime-*.zip'))
    assert len(built) == 1, built
    archive = artifact / built[0].name
    shutil.copy2(built[0], archive)
    with zipfile.ZipFile(archive) as z:
        members = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
    mismatched = [n for n, h in members.items() if source_sha(n) != h]
    assert not mismatched, mismatched
    assert not any(n.lower().endswith('.pdb') or '__pycache__' in n for n in members)
    lanes = {}
    for lane in ('vulkan', 'opengl'):
        env = profile_env(profiles / lane)
        rows.append(run(f'install-{lane}', ['install-file', '--repo', 'user_default', '--enable', str(archive)],
                        env, logs))
        installed = Path(env['BLENDER_USER_EXTENSIONS']) / 'user_default' / 'warp_sph_realtime'
        diff = [n for n, h in members.items() if not (installed / n).is_file() or sha(installed / n) != h]
        assert not diff, (lane, diff)
        lanes[lane] = {'env': {k: v for k, v in env.items() if k.startswith('BLENDER_USER_')},
                       'installed_dir': str(installed), 'members_match_archive': True}
    receipt = {'blender': EXE, 'blender_sha256': sha(Path(EXE)), 'profile_identity': identity, 'python': sys.version,
               'archive': str(archive.relative_to(ROOT)), 'archive_sha256': sha(archive),
               'members': members, 'members_match_source': True, 'checks': rows, 'lanes': lanes}
    (OUT / 'package.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('archive', 'archive_sha256')}, indent=2), len(members), 'members')


if __name__ == '__main__':
    main()
