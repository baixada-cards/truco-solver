"""Verify every declared engine dependency matches engine.lock.json."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_MANIFEST = ROOT / "Cargo.toml"
LOCK = ROOT / "engine.lock.json"
CRATE_MANIFESTS = [
    ROOT / "crates/truco-policy-format/Cargo.toml",
    ROOT / "crates/truco-solver/Cargo.toml",
]


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    root = tomllib.loads(ROOT_MANIFEST.read_text(encoding="utf-8"))
    dependency = root["workspace"]["dependencies"]["truco-engine"]

    failures: list[str] = []
    if lock.get("format") != "baixada-engine-lock/v1":
        failures.append("engine.lock.json has an unsupported format")
    if dependency.get("git") != lock.get("repository"):
        failures.append("workspace engine Git URL differs from engine.lock.json")
    if dependency.get("rev") != lock.get("revision"):
        failures.append("workspace engine revision differs from engine.lock.json")

    for manifest_path in CRATE_MANIFESTS:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        declared = manifest["dependencies"].get("truco-engine")
        if declared != {"workspace": True}:
            failures.append(
                f"{manifest_path.relative_to(ROOT)} must inherit truco-engine "
                "from workspace dependencies"
            )

    cargo_lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    expected_source = (
        f'git+{lock["repository"]}?rev={lock["revision"]}#{lock["revision"]}'
    )
    if expected_source not in cargo_lock:
        failures.append("Cargo.lock does not contain the exact engine revision")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print(f"Engine dependency is locked to {lock['revision']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
