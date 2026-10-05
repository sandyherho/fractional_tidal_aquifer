"""Regenerate every figure, animation, data file, and report.

Figures 5 to 8 cache their expensive runs in outputs/cache, which the
animations and the report script read; an interrupted run resumes where
it stopped.  Delete outputs/cache to recompute everything.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_schematic.py", "fig01_capacity.py", "fig02_ratio.py",
         "fig03_discriminant.py", "fig04_bound.py", "fig05_timedomain.py",
         "fig06_nonlinear.py", "fig07_information.py",
         "fig08_verification.py", "anim01_breathing.py", "anim02_ratio.py",
         "anim03_watertable.py", "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
