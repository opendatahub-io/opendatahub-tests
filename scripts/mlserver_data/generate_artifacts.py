#!/usr/bin/env python3
"""Generate MLServer model artifacts for multi-model serving tests.

Trains deterministic, tiny models on the Iris dataset and writes them
alongside the version-controlled model-settings.json files into an
output directory ready for S3 upload.

Dependencies (install via pip):
    scikit-learn numpy joblib xgboost lightgbm skl2onnx onnxruntime

Usage:
    python scripts/mlserver_data/generate_artifacts.py --output-dir /tmp/mlserver_models

The output mirrors the six S3 prefixes consumed by the test fixtures:
    mlserver/model_repository/   -- 10-model happy path
    mlserver/multi_version/      -- shared-model v1.0 + v2.0
    mlserver/one_failed_model/   -- 9 valid + 1 bad artifact ref
    mlserver/malformed_settings/ -- 2 valid + broken JSON
    mlserver/unsupported_impl/   -- 2 valid + custom.MyModel
    mlserver/missing_artifact/   -- 2 valid + nonexistent URI
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_DIR = SCRIPT_DIR / "model_settings"

RANDOM_STATE = 42


def _train_sklearn() -> DecisionTreeClassifier:
    X, y = load_iris(return_X_y=True)
    clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    clf.fit(X=X, y=y)
    return clf


def _save_sklearn(model: DecisionTreeClassifier, path: Path) -> None:
    joblib.dump(value=model, filename=path)


def _save_xgboost(path: Path) -> None:
    import xgboost as xgb

    X, y = load_iris(return_X_y=True)
    dtrain = xgb.DMatrix(data=X, label=y)
    params = {
        "max_depth": 3,
        "num_class": 3,
        "objective": "multi:softprob",
        "seed": RANDOM_STATE,
    }
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)
    bst.save_model(fname=str(path))


def _save_lightgbm(path: Path) -> None:
    import lightgbm as lgb

    X, y = load_iris(return_X_y=True)
    train_data = lgb.Dataset(data=X, label=y)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "max_depth": 3,
        "num_leaves": 8,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }
    bst = lgb.train(params=params, train_set=train_data, num_boost_round=10)
    bst.save_model(filename=str(path))


def _save_onnx(path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    clf = _train_sklearn()
    initial_types = [("predict", FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(
        model=clf,
        initial_types=initial_types,
        target_opset=13,
    )
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


# Map implementation -> (artifact filename, generator function)
ARTIFACT_GENERATORS: dict[str, tuple[str, Any]] = {
    "mlserver_sklearn.SKLearnModel": ("model.joblib", lambda path: _save_sklearn(model=_train_sklearn(), path=path)),
    "mlserver_xgboost.XGBoostModel": ("model.json", _save_xgboost),
    "mlserver_lightgbm.LightGBMModel": ("model.txt", _save_lightgbm),
    "mlserver_onnx.OnnxModel": ("model.onnx", _save_onnx),
}


def generate(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_dir in sorted(SETTINGS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue

        dest_scenario = (output_dir / scenario_dir.name).resolve()
        dest_scenario.relative_to(output_dir)  # noqa: FCN001
        dest_scenario.mkdir(parents=True, exist_ok=True)

        for model_dir in sorted(scenario_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            dest_model = (dest_scenario / model_dir.name).resolve()
            dest_model.relative_to(output_dir)  # noqa: FCN001
            dest_model.mkdir(parents=True, exist_ok=True)

            settings_file = model_dir / "model-settings.json"
            if not settings_file.exists():
                continue

            # Copy model-settings.json
            settings_dest = (dest_model / "model-settings.json").resolve()
            settings_dest.relative_to(output_dir)  # noqa: FCN001
            shutil.copy2(src=settings_file, dst=settings_dest)

            # Parse settings to determine which artifact to generate
            raw = settings_file.read_text()
            try:
                settings = json.loads(raw)
            except json.JSONDecodeError:
                # Intentionally malformed settings (e.g. bad-model)
                print(f"  {scenario_dir.name}/{model_dir.name}: malformed JSON, skipping artifact")
                continue

            impl = settings.get("implementation", "")
            uri = settings.get("parameters", {}).get("uri", "")

            # Skip models that intentionally reference missing artifacts
            if "nonexistent" in uri:
                print(f"  {scenario_dir.name}/{model_dir.name}: intentionally missing artifact, skipping")
                continue

            # Skip unsupported implementations (e.g. custom.MyModel)
            if impl not in ARTIFACT_GENERATORS:
                print(f"  {scenario_dir.name}/{model_dir.name}: unsupported impl '{impl}', skipping artifact")
                continue

            artifact_name, generator_fn = ARTIFACT_GENERATORS[impl]
            artifact_path = (dest_model / artifact_name).resolve()
            artifact_path.relative_to(output_dir)  # noqa: FCN001
            generator_fn(path=artifact_path)
            print(f"  {scenario_dir.name}/{model_dir.name}: {artifact_name}")


def verify(output_dir: Path) -> None:
    """Quick sanity check: count artifacts per scenario."""
    for scenario_dir in sorted(output_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        model_dirs = [d for d in scenario_dir.iterdir() if d.is_dir()]
        settings_count = sum(1 for d in model_dirs if (d / "model-settings.json").exists())
        artifact_count = sum(
            1
            for d in model_dirs
            for f in d.iterdir()
            if f.suffix in (".joblib", ".json", ".txt", ".onnx") and f.name != "model-settings.json"
        )
        print(f"  {scenario_dir.name}: {settings_count} settings, {artifact_count} artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the generated model tree into.",
    )
    args = parser.parse_args()

    print("Generating model artifacts...")
    generate(output_dir=args.output_dir)
    print("\nVerification:")
    verify(output_dir=args.output_dir)
    print(
        f"\nDone. Upload with:\n"
        f"  python scripts/mlserver_data/upload_to_s3.py"
        f" --source-dir {args.output_dir} --bucket <BUCKET>"
    )


if __name__ == "__main__":
    main()
