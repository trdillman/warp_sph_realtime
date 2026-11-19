"""Use CPU-only pytest by default on workstations with a CUDA device."""
import os

if os.environ.get("WARPSPH_TEST_CUDA") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
