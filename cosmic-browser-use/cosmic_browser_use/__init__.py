"""cosmic_browser_use — programmatic API for the Cosmic-OS browser specialist.

The heavyweight modules (main.py, browser_controller.py, orchestrator.py,
browser_memory/) live in this directory but intentionally keep their original
top-level import names. This package bootstraps sys.path so a consumer (the
Cosmic-OS browser agent specialist) can simply:

    from cosmic_browser_use.api import run_goal

An editable checkout is expected (BROWSER_USE_HOME or the parent of this
package's directory is added to sys.path).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_import_path() -> Path:
    """Make the package's parent directory importable so `main`, the
    `browser_memory` package, etc. resolve."""
    here = Path(__file__).resolve().parent      # .../cosmic-browser-use/cosmic_browser_use
    home = here.parent                          # .../cosmic-browser-use
    env_home = os.getenv("BROWSER_USE_HOME", "").strip()
    if env_home:
        env_home_path = Path(env_home).expanduser().resolve()
        if env_home_path.is_dir() and (env_home_path / "main.py").is_file():
            home = env_home_path
    home_str = str(home)
    if home_str not in sys.path:
        sys.path.insert(0, home_str)
    return home


_HOME = _bootstrap_import_path()

__all__ = ["run_goal", "get_repo_home"]

def get_repo_home() -> Path:
    """Directory containing main.py / browser_memory for this checkout."""
    return _HOME
