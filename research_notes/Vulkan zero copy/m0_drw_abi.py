"""Offline exact-image context ownership ABI discovery; no target execution."""
from pathlib import Path
p=Path(__file__).with_name('m0_context_abi.py')
source=p.read_text().replace("patterns=['*wm_window_reset_drawable*','*wm_window_set_drawable*','*GPU_context_active_set*','*wm_window_make_drawable*']", "patterns=['*DRW_gpu_context*','*DST*','*GHOST_GetDrawingContext*']")
source=source.replace("'context-abi.json'", "'drw-context-abi.json'")
exec(compile(source,str(p),'exec'))
