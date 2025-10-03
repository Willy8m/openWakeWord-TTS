from openwakeword.train import main, Model, convert_onnx_to_tflite
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phonetic variations for a wakeword.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with augmented audio features")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save model")
    parser.add_argument("--config_dir", type=str, required=True, help="Folder with training config")
    args = parser.parse_args()

    args.generate_clips = False
    args.augment_clips = False
    args.train_model = True
    args.training_config = os.path.join(args.config_dir, os.listdir(args.config_dir)[0])
    args.overwrite = False

    main(Model, convert_onnx_to_tflite, args)