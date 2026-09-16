"""Engineering-only flock obtain/conflict/release check; no fit/data access."""
from __future__ import annotations
import fcntl, json, os, subprocess, sys
from pathlib import Path

RUN = Path(__file__).resolve().parent
LOCK = RUN / "lock_selftest.lock"
if LOCK.exists():
    LOCK.unlink()
child = "import fcntl,sys; h=open(sys.argv[1], 'a+');\ntry: fcntl.flock(h.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: raise SystemExit(17)\nraise SystemExit(0)"
parent = LOCK.open("a+")
fctl_parent = True
fcntl.flock(parent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
conflict = subprocess.run([sys.executable, "-c", child, str(LOCK)], check=False)
parent.close()
second = LOCK.open("a+")
fctl_second = True
fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
second.close()
LOCK.unlink()
result = {"schema": "p9016-flock-selftest-v1", "status": "PASS" if conflict.returncode == 17 else "FAIL",
          "first_acquire": fctl_parent, "conflict_return_code": conflict.returncode,
          "release_then_reacquire": fctl_second, "reference_opened": False, "synthetic_fits": 0}
(RUN / "lock_selftest.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
raise SystemExit(0 if result["status"] == "PASS" else 1)
