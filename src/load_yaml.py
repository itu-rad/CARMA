import os
from pathlib import Path
import yaml

def load_yaml(default_name="config.yaml"):
    # 1) Prefer an explicit env var if you set it
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}
        raise FileNotFoundError(f"CONFIG_PATH points to a missing file: {p}")

    # 2) Try next to this script (works even if CWD is different)
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [
        script_dir / default_name,   # ./config.yaml beside the script
        Path.cwd() / default_name,   # ./config.yaml in current working dir
    ]

    for p in candidates:
        if p.exists():
            return yaml.safe_load(p.read_text()) or {}

    # Helpful error with diagnostics
    tried = "\n  - ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        "Could not find config.yaml. I looked at:\n  - " + tried +
        f"\nCurrent working directory: {Path.cwd().resolve()}"
    )