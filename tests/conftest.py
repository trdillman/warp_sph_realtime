"""Use CPU-only pytest by default on workstations with a CUDA device, and the pinned bundled SDK."""
import os
import sys
from pathlib import Path

if os.environ.get("WARPSPH_TEST_CUDA") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# The pinned Blender Vulkan Interop SDK is a git submodule (private repository). Host tests use it
# directly so they never depend on a separately installed copy. Fetch it with:
#   git submodule update --init vendor/blender-vulkan-interop
VENDORED_SDK_SRC = Path(__file__).resolve().parents[1] / "vendor" / "blender-vulkan-interop" / "src"
if (VENDORED_SDK_SRC / "blender_vulkan_interop" / "__init__.py").is_file():
    sys.path.insert(0, str(VENDORED_SDK_SRC))


def pytest_report_header(config):
    if (VENDORED_SDK_SRC / "blender_vulkan_interop" / "__init__.py").is_file():
        return f"shared SDK: bundled copy at {VENDORED_SDK_SRC}"
    return ("shared SDK: MISSING. SDK tests will fail; run "
            "`git submodule update --init vendor/blender-vulkan-interop` (needs access to the private repo)")
