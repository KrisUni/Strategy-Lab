"""
Local filesystem persistence for named strategy configurations.
Storage: ~/.strategy_lab/strategies/{slug}.json
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def _strategies_dir() -> Path:
    d = Path.home() / ".strategy_lab" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "strategy"


def _path_for_name(name: str) -> Path | None:
    for f in _strategies_dir().glob("*.json"):
        try:
            if json.loads(f.read_text()).get("name") == name:
                return f
        except Exception:
            pass
    return None


def save_strategy(name: str, params: dict, execution: dict) -> Path:
    existing = _path_for_name(name)
    if existing:
        path = existing
    else:
        slug = _slug(name)
        path = _strategies_dir() / f"{slug}.json"
        i = 2
        while path.exists():
            path = _strategies_dir() / f"{slug}_{i}.json"
            i += 1
    path.write_text(
        json.dumps(
            {
                "name": name,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": SCHEMA_VERSION,
                "params": params,
                "execution": execution,
            },
            indent=2,
            default=str,
        )
    )
    return path


def load_strategy(name: str) -> tuple[dict, dict]:
    path = _path_for_name(name)
    if path is None:
        raise FileNotFoundError(f"Strategy {name!r} not found")
    data = json.loads(path.read_text())
    return data.get("params", {}), data.get("execution", {})


def list_strategies() -> list[dict]:
    results = []
    for f in _strategies_dir().glob("*.json"):
        try:
            data = json.loads(f.read_text())
            results.append(
                {
                    "name": data.get("name", f.stem),
                    "saved_at": data.get("saved_at", ""),
                    "file": f,
                }
            )
        except Exception:
            pass
    return sorted(results, key=lambda x: x["saved_at"], reverse=True)


def delete_strategy(name: str) -> None:
    path = _path_for_name(name)
    if path is None:
        raise FileNotFoundError(f"Strategy {name!r} not found")
    path.unlink()
