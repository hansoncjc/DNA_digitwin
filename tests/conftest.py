"""
Test setup: make the repo importable without MC-DFM or HOOMD.

``scattering.py`` imports MC-DFM's ``Scattering_Simulator`` at module level
and ``simulation.py`` imports ``hoomd``. Neither is needed by the pure
functions under test, so each is replaced by an empty stub module only when
the real package cannot be imported.
"""
import importlib
import signal
import sys
import types
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_if_missing(name, submodules=()):
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    root = types.ModuleType(name)
    root.__path__ = []
    sys.modules[name] = root
    for sub in submodules:
        mod = types.ModuleType(f"{name}.{sub}")
        sys.modules[f"{name}.{sub}"] = mod
        setattr(root, sub, mod)


_stub_if_missing("Scattering_Simulator", submodules=("pairwise_method",))
_stub_if_missing("hoomd", submodules=("md",))



@pytest.fixture(autouse=True)
def _launcher_registry_isolated():
    """Each test starts and ends with an empty launcher job list, no shutdown
    flag and pytest's own SIGTERM handler. Fake Slurm ids registered by a test
    must never reach the atexit hook: on the cluster it would run the real
    scancel on them."""
    from parallel import submit_parallel as sp

    def reset():
        with sp._LIVE_LOCK:
            sp._LIVE.clear()
        sp._SHUTDOWN = False
        sp._SIGNAL_HANDLERS_INSTALLED = False

    handler = signal.getsignal(signal.SIGTERM)
    reset()
    yield
    reset()
    signal.signal(signal.SIGTERM, handler)
