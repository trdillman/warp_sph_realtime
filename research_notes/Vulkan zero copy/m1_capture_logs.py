"""Reuse retained-log copying policy, with a separate M1 evidence root."""
from pathlib import Path
p=Path(__file__).with_name('m0_capture_logs.py')
source=p.read_text().replace('20261017-contained-m0','20261018-m1')
exec(compile(source,str(p),'exec'))
