#!/usr/bin/env python3
import os
import re
import argparse
import yaml
from pathlib import Path


def get_next_data_version(models_dir: Path, wakeword: str) -> int:
    """
    Finds the highest <data_version> from folders named <wakeword>_<data_version>
    in models_dir. Returns next available version.
    """
    max_version = -1
    if models_dir.exists():
        for entry in models_dir.iterdir():
            if entry.is_dir():
                match = re.match(rf"^{wakeword}_(\d+)$", entry.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))
    return max_version + 1


def get_next_model_version(models_dir: Path, wakeword: str, data_version: int) -> int:
    """
    Finds the highest <model_version> from .onnx files named <wakeword>_<data_version>_<model_version>.onnx
    in models_dir. Returns next available version.
    """
    max_version = -1
    if models_dir.exists():
        for entry in models_dir.iterdir():
            if entry.is_file() and entry.suffix == ".onnx":
                match = re.match(rf"^{wakeword}_{data_version}_(\d+)\.onnx$", entry.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))
    return max_version + 1


def generate_yaml(base_path: str, wakeword: str, data_version: int, model_version: int) -> dict:
    return {
        "wakeword": f"{wakeword}",
        "output_dir": f"{base_path}models/{wakeword}",
        "data_folder": f"{wakeword}_{data_version}",
        "model_name": f"{wakeword}_{data_version}_{model_version}",
        "augmentation_batch_size": 128,
        "augmentation_rounds": 500,
        "rir_paths": [
            f"{base_path}data/train/mit_rirs"
        ],
        "background_paths": [
            f"{base_path}data/train/NEGATIVE/wham_noise/tr",
            f"{base_path}data/train/NEGATIVE/DEMAND"
        ],
        "background_paths_duplication_rate": [1, 1],
        "model_type": "dnn",
        "hidden_layers": 5,
        "layer_size": 64,
        "steps": 15000,
        "max_negative_weight": 100,
        "batch_n_per_class": {
            "positive": 64,
            "adversarial_negative": 64,
            "ACAV100M_sample": 512,
            "fma": 512,
            "podcasts": 512,
            "tv3_train": 512,
            "wham_tr": 64
        },
        "feature_data_files": {
            "ACAV100M_sample": f"{base_path}data/train/NEGATIVE/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "fma": f"{base_path}data/train/NEGATIVE/fma_large.npy",
            "podcasts": f"{base_path}data/train/NEGATIVE/openwakeword_features_podcasts_10000h_ca-es_ca_es-es.npy",
            "tv3_train": f"{base_path}data/train/NEGATIVE/tv3_train.npy",
            "wham_tr": f"{base_path}data/train/NEGATIVE/wham_tr.npy"
        },
        "false_positive_validation_data_path": f"{base_path}data/validation/validation_set_features.npy",
        "target_accuracy": 0.85,
        "target_recall": 0.25,
        "target_false_positives_per_hour": 0.2
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a YAML config for wakeword training.")
    parser.add_argument("--base_path", required=True, help="Base path for data and models")
    parser.add_argument("--wakeword", required=True, help="Wakeword name")
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    models_dir = base_path / "models" / args.wakeword

    data_version = get_next_data_version(models_dir, args.wakeword)
    model_version = get_next_model_version(models_dir, args.wakeword, data_version)

    config = generate_yaml(str(base_path), args.wakeword, data_version, model_version)

    # Save YAML file
    yaml_filename = f"{args.wakeword}_{data_version}_{model_version}.yaml"
    yaml_path = models_dir / yaml_filename
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ YAML config generated: {yaml_path}")


if __name__ == "__main__":
    main()
