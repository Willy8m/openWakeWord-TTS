# pipeline.py
from pathlib import Path
import subprocess
import os
import sys
import logging
from datetime import datetime
import argparse

# -----------------------------
# Configuration
# -----------------------------
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = "F:/data"
LOCALE = "es-es"

# Enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUTS_DIR, "logs", "pipeline.log"))
    ]
)

# -----------------------------
# Helper Functions
# -----------------------------
def run_step(script, **kwargs):
    """Run a Python script with named arguments."""
    cmd = ["uv", "run", os.path.join(SCRIPTS_DIR, script)]
    for k, v in kwargs.items():
        cmd.append(f"--{k}={v}")
    logging.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def create_folders(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

# -----------------------------
# Pipeline Main
# -----------------------------
def pipeline(wakewords):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for wakeword in wakewords:
        logging.info(f"=== Starting pipeline for wakeword: {wakeword} ===")
        
        # Define output folders for this wakeword
        config_dir = os.path.join(OUTPUTS_DIR, wakeword, "config", timestamp)
        txt_dir = os.path.join(OUTPUTS_DIR, wakeword, "txt", timestamp)
        audio_dir = os.path.join(OUTPUTS_DIR, wakeword, "audio", timestamp)
        aug_dir = os.path.join(OUTPUTS_DIR, wakeword, "augmented_audio", timestamp)
        model_dir = os.path.join(OUTPUTS_DIR, wakeword, "models", timestamp)
        
        create_folders(config_dir, txt_dir, audio_dir, aug_dir, model_dir)

        # Step 0: Generate training config file
        run_step("generate_config.py", wakeword=wakeword, output_folder=config_dir, data_dir=DATA_DIR)
        
        # Step 1: Generate text (OpenAI)
        run_step("generate_text.py", output_folder=txt_dir, locale=LOCALE)
        
        # Step 2: TTS conversion
        run_step("tts.py", input_folder=txt_dir, output_folder=audio_dir)
        
        # Step 3: Data augmentation
        run_step("augment.py", input_folder=audio_dir, output_folder=aug_dir, config_dir=config_dir)
        
        # Step 4: Model training
        run_step("train.py", input_folder=aug_dir, output_folder=model_dir, config_dir=config_dir)
        
        # # Step 5: Upload to blob storage
        # run_step("upload_blob.py", model_folder=model_dir, wakeword=wakeword)
        
        logging.info(f"=== Pipeline finished for wakeword: {wakeword} ===\n")

# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wakeword", help="Single wakeword")
    parser.add_argument("--wakewords", nargs="+", help="Multiple wakewords")
    args = parser.parse_args()

    if args.wakeword:
        pipeline([args.wakeword])
    elif args.wakewords:
        pipeline(args.wakewords)
    else:
        print("❌ You must provide --wakeword or --wakewords")
        sys.exit(1)
