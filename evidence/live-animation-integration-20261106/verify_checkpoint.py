"""Read-only checkpoint verification; JSON result is captured by the caller."""
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(r"C:\Users\Tyler\Downloads\WarpSPH-Live-GN-Animation-Cross-Platform-Checkpoint-2026-11-06.zip")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


with zipfile.ZipFile(ARCHIVE) as archive:
    manifest = json.loads(archive.read("MANIFEST.json"))
    source = zipfile.ZipFile(io.BytesIO(archive.read("source/committed-source.zip")))
    checkpoint = manifest["final_evidence_source_commit"]
    source_bad = [name for name, digest in manifest["source_file_sha256"].items()
                  if hashlib.sha256(source.read(name)).hexdigest() != digest
                  or source.read(name) != git("show", f"{checkpoint}:{name}")]
    asset_bad = [name for name, digest in manifest["asset_sha256"].items()
                 if hashlib.sha256(archive.read(name)).hexdigest() != digest]
    result = {
        "archive": str(ARCHIVE),
        "archive_sha256": hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest(),
        "checkpoint": checkpoint,
        "main": git("rev-parse", "main").decode().strip(),
        "origin_main": git("rev-parse", "origin/main").decode().strip(),
        "main_checkpoint_left_right": git("rev-list", "--left-right", "--count", f"origin/main...{checkpoint}").decode().strip(),
        "source_files_verified": len(manifest["source_file_sha256"]),
        "assets_verified": len(manifest["asset_sha256"]),
        "source_mismatches": source_bad,
        "asset_mismatches": asset_bad,
        "checkpoint_limits": manifest["limits"],
    }
    print(json.dumps(result, indent=2))
    assert not source_bad and not asset_bad
