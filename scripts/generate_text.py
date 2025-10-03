import os
import argparse
from dotenv import load_dotenv
from openai import AzureOpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")  # e.g. "gpt-35-turbo" or "gpt-4o-mini"

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_MODEL):
    raise ValueError("Missing Azure OpenAI configuration in .env (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_MODEL)")

# ----------------------------
# Azure OpenAI Client
# ----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,  # adapt if needed
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


def generate_variations(wakeword: str):
    """Generate positive and negative phonetic variations for a wakeword."""
    
    positive_prompt = f"""
    Generate a list of at least 15 positive phonetic variations for the wakeword "{wakeword}".
    These should sound similar but represent natural pronunciation variations, accents, or small distortions.
    Return them as a plain list, one per line, without numbering.
    """

    negative_prompt = f"""
    Generate a list of at least 15 adversarial phonetic phrases that sound similar to "{wakeword}" 
    but are clearly different words or short phrases. 
    The goal is to confuse a speech recognition model. 
    Return them as a plain list, one per line, without numbering.
    """

    # Call for positive variations
    pos_response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[{"role": "system", "content": "You are a linguistics and phonetics expert."},
                  {"role": "user", "content": positive_prompt}],
        temperature=0.8
    )

    # Call for negative variations
    neg_response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL,
        messages=[{"role": "system", "content": "You are a linguistics and phonetics expert."},
                  {"role": "user", "content": negative_prompt}],
        temperature=0.8
    )

    positives = pos_response.choices[0].message.content.strip().splitlines()
    negatives = neg_response.choices[0].message.content.strip().splitlines()

    return positives, negatives


def main(output_folder: str, wakeword: str):
    os.makedirs(output_folder, exist_ok=True)

    pos_file = os.path.join(output_folder, "pos.txt")
    neg_file = os.path.join(output_folder, "neg.txt")

    positives, negatives = generate_variations(wakeword)

    with open(pos_file, "w", encoding="utf-8") as f:
        f.write("\n".join(positives))

    with open(neg_file, "w", encoding="utf-8") as f:
        f.write("\n".join(negatives))

    print(f"[+] Written {len(positives)} positive variations to {pos_file}")
    print(f"[+] Written {len(negatives)} negative variations to {neg_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phonetic variations for a wakeword.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save pos.txt and neg.txt")
    parser.add_argument("--wakeword", type=str, required=True, help="Target wakeword")
    args = parser.parse_args()

    main(args.output_folder, args.wakeword)
