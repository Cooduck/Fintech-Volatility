"""Persist and verify historical model bundles and their input identities."""

import hashlib
import json
from pathlib import Path

import joblib

from config import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(out):
    return json.loads((out / "frozen_manifest.json").read_text(encoding="utf8"))


def source_hashes(paths):
    names = {Path(path).relative_to(ROOT).as_posix() for path in paths}
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").glob("*.py"))
    return {name: digest(ROOT / name) for name in sorted(names)}


def save_model(path, models):
    temporary = path.with_suffix(".tmp")
    joblib.dump(models, temporary)
    temporary.replace(path)
    return digest(path)


def save_manifest(out, manifest):
    path = out / "frozen_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    temporary.replace(path)


def publish_bundle(out, models, metadata, paths, *, baseline=None):
    manifest = dict(metadata, data_and_code=source_hashes(paths))
    manifest["model_sha256"] = save_model(out / "frozen_models.joblib", models)
    if baseline is not None:
        manifest["baseline_sha256"] = save_model(out / "frozen_baseline.joblib", baseline)
    save_manifest(out, manifest)


def load_models(out, *, baseline=False):
    """Load a locally published bundle after checking its artifact digest."""
    manifest = read_manifest(out)
    name = "baseline" if baseline else "models"
    key = "baseline_sha256" if baseline else "model_sha256"
    path = out / f"frozen_{name}.joblib"
    if digest(path) != manifest[key]:
        raise ValueError(f"Model artifact differs from its manifest: {path}")
    return joblib.load(path)
