import os
import re
import argparse
import yaml
from pathlib import Path

CODE_VERSION = int(os.getenv("CODE_VERSION"))

def get_next_model_version(config_dir: Path, wakeword: str, data_version: int, CODE_VERSION: int) -> int:
    """
    Finds the highest <model_version> from .yaml files named:
      <wakeword>_v<data_version>_<CODE_VERSION>_<model_version>.yaml
    in config_dir and its subdirectories. Returns next available version.
    """
    max_version = -1
    if config_dir.exists():
        for entry in config_dir.rglob("*.yaml"):  # recursive search
            if entry.is_file():
                match = re.match(rf"^{wakeword}_v{data_version}_{CODE_VERSION}_(\d+)\.yaml$", entry.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))
    return max_version + 1


def get_next_data_version(config_dir: Path, wakeword: str) -> int:
    """
    Finds the highest <data_version> from .yaml files named:
      <wakeword>_v<data_version>_*.yaml
    in config_dir and its subdirectories. Returns next available version.
    """
    max_version = 0
    if config_dir.exists():
        for entry in config_dir.rglob("*.yaml"):  # recursive search
            if entry.is_file():
                match = re.match(rf"^{wakeword}_v(\d+)_.*\.yaml$", entry.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))
    return max_version + 1


def generate_training_config_yaml(wakeword_dir, data_dir: str, wakeword: str, data_version: int, model_version: int) -> dict:
    return {
        "wakeword": f"{wakeword}",
        "output_dir": f"{wakeword_dir}models/{wakeword}",
        "data_folder": f"{wakeword}_v{data_version}_{CODE_VERSION}_X",
        "model_name": f"{wakeword}_v{data_version}_{CODE_VERSION}_{model_version}",
        "augmentation_batch_size": 128,
        "augmentation_rounds": 500,
        "rir_paths": [
            f"{data_dir}train/mit_rirs"
        ],
        "background_paths": [
            f"{data_dir}train/NEGATIVE/wham_noise/tr",
            f"{data_dir}train/NEGATIVE/DEMAND"
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
            "ACAV100M_sample": f"{data_dir}train/NEGATIVE/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            "fma": f"{data_dir}train/NEGATIVE/fma_large.npy",
            "podcasts": f"{data_dir}train/NEGATIVE/openwakeword_features_podcasts_10000h_ca-es_ca_es-es.npy",
            "tv3_train": f"{data_dir}train/NEGATIVE/tv3_train.npy",
            "wham_tr": f"{data_dir}train/NEGATIVE/wham_tr.npy"
        },
        "false_positive_validation_data_path": f"{data_dir}validation/validation_set_features.npy",
        "target_accuracy": 0.85,
        "target_recall": 0.25,
        "target_false_positives_per_hour": 0.2
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a YAML config for wakeword training.")
    parser.add_argument("--wakeword", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--data_version", required=False, type=int, default=1, help=f"Version of the positive and adversarial data")
    parser.add_argument("--data_dir", required=False, type=Path, default="./data/")
    args = parser.parse_args()

    wakeword = args.wakeword
    output_folder = Path(args.outputs_dir)
    data_dir = args.data_dir

    wakeword_dir = output_folder.parent.parent
    timestamp = output_folder.parts[-1]

    data_version = args.data_version if args.data_version else get_next_data_version(wakeword_dir, wakeword)
    model_version = 0 if (data_version != args.data_version) else get_next_model_version(wakeword_dir, wakeword, data_version, CODE_VERSION)

    config = generate_training_config_yaml(
        wakeword_dir=wakeword_dir,
        data_dir=data_dir, 
        wakeword=wakeword, 
        data_version=data_version, 
        model_version=model_version
    )

    # Save YAML file
    yaml_filename = f"{wakeword}_v{data_version}_{CODE_VERSION}_{model_version}.yaml"
    yaml_path = output_folder / yaml_filename

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ YAML config generated: {yaml_path}")


if __name__ == "__main__":
    main()
