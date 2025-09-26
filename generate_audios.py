import os
import io
import argparse
from dotenv import load_dotenv
from pydub import AudioSegment, effects
import azure.cognitiveservices.speech as speechsdk

from helpers.create_ssml import create_ssml

"""This script generates audio from text through the azure TTS api.
You will need a subscription key to be set up on .env file (see README setup & usage).
For config params: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
"""

AUDIOS_OUTPUT_FOLDER = r"output_audios"
WORDS_FOLDER = r"input_words"

# === CONFIG ===
VOICES = ["ca-ES-EnricNeural", "ca-ES-AlbaNeural", "ca-ES-JoanaNeural", "es-AR-ElenaNeural", "es-AR-TomasNeural", "es-BO-SofiaNeural", "es-BO-MarceloNeural", "es-CL-CatalinaNeural", "es-CL-LorenzoNeural", "es-CO-SalomeNeural", "es-CO-GonzaloNeural", "es-CR-MariaNeural", "es-CR-JuanNeural", "es-CU-BelkysNeural", "es-CU-ManuelNeural", "es-DO-RamonaNeural", "es-DO-EmilioNeural", "es-EC-AndreaNeural", "es-EC-LuisNeural", "es-ES-ElviraNeural", "es-ES-AlvaroNeural", "es-ES-AbrilNeural", "es-ES-ArnauNeural", "es-ES-DarioNeural", "es-ES-EliasNeural", "es-ES-EstrellaNeural", "es-ES-IreneNeural", "es-ES-LaiaNeural", "es-ES-LiaNeural", "es-ES-NilNeural", "es-ES-SaulNeural", "es-ES-TeoNeural", "es-ES-TrianaNeural", "es-ES-VeraNeural", "es-ES-XimenaNeural", "es-ES-ArabellaMultilingualNeural", "es-ES-IsidoraMultilingualNeural", "es-ES-TristanMultilingualNeural", "es-ES-XimenaMultilingualNeural", "es-GQ-TeresaNeural", "es-GQ-JavierNeural", "es-GT-MartaNeural", "es-GT-AndresNeural", "es-HN-KarlaNeural", "es-HN-CarlosNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-MX-BeatrizNeural", "es-MX-CandelaNeural", "es-MX-CarlotaNeural", "es-MX-CecilioNeural", "es-MX-GerardoNeural", "es-MX-LarissaNeural", "es-MX-LibertoNeural", "es-MX-LucianoNeural", "es-MX-MarinaNeural", "es-MX-NuriaNeural", "es-MX-PelayoNeural", "es-MX-RenataNeural", "es-MX-YagoNeural", "es-NI-YolandaNeural", "es-NI-FedericoNeural", "es-PA-MargaritaNeural", "es-PA-RobertoNeural", "es-PE-CamilaNeural", "es-PE-AlexNeural", "es-PR-KarinaNeural", "es-PR-VictorNeural", "es-PY-TaniaNeural", "es-PY-MarioNeural", "es-SV-LorenaNeural", "es-SV-RodrigoNeural", "es-US-PalomaNeural", "es-US-AlonsoNeural"]
RATES = [0]         # e.g., [-20, 0, 20]
PITCHES = [0]       # e.g., [-10, 0, 10]
STYLES = [None]     # e.g., ["cheerful", "sad"]
ROLES = [None]      # e.g., ["YoungAdultFemale", "SeniorMale"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wakeword", required=True, type=str, help="wake word (separated by hyphens)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (only first voice, rate=0, pitch=0)")
    parser.add_argument("--silence_len", default=1000, type=int, help="(in ms) increase value if wakeword is being split in the middle. default: 1000ms")
    parser.add_argument("--silence_padding", default=500, type=int, help="(in ms) increase if wakeword is being trimmed too soon at the end. default: 500ms")
    parser.add_argument("--silence_thresh", default=-40, type=int, help="(in dBFS) anything quieter than this will be considered silence. default: -40dBFS")
    args = parser.parse_args()

    wakeword = args.wakeword
    isTest = args.test
    silence_len = args.silence_len
    silence_padding = args.silence_padding
    silence_thresh = args.silence_thresh
 
    print("Setting up TTS...")
    speech_config = setup_tts()

    print(f"Loading words from .txt files... (at {WORDS_FOLDER})")
    neg = read_to_list(os.path.join(WORDS_FOLDER, wakeword, f"neg.txt"))
    pos = read_to_list(os.path.join(WORDS_FOLDER, wakeword, f"pos.txt"))

    print(f"Preparing output folders... (at {AUDIOS_OUTPUT_FOLDER})")
    if isTest:
        dirs = {
            "pos": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "tts_test", "pos"),
            "neg": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "tts_test", "neg")
        }
    else:
        dirs = {
            "pos_train": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "positive_train"),
            "pos_test": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "positive_test"),
            "neg_train": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "negative_train"),
            "neg_test": os.path.join(AUDIOS_OUTPUT_FOLDER, wakeword, "negative_test"),
        }

    print("Starting the conversion...")
    if isTest:
        generate_audios(pos, [VOICES[0]], speech_config, dirs["pos"], isTest, args.silence_len, silence_padding, silence_thresh)
        generate_audios(neg, [VOICES[0]], speech_config, dirs["neg"], isTest, args.silence_len, silence_padding, silence_thresh)
    else:
        train_voices, test_voices = train_test_split(VOICES)
        generate_audios(pos, train_voices, speech_config, dirs["pos_train"], isTest, silence_len, silence_padding, silence_thresh)
        generate_audios(pos, test_voices,  speech_config, dirs["pos_test"],  isTest, silence_len, silence_padding, silence_thresh)
        generate_audios(neg, train_voices, speech_config, dirs["neg_train"], isTest, silence_len, silence_padding, silence_thresh)
        generate_audios(neg, test_voices,  speech_config, dirs["neg_test"],  isTest, silence_len, silence_padding, silence_thresh)


def setup_tts():
    load_dotenv()
    subscription_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not (subscription_key or region):
        raise ValueError("Missing Azure credentials in environment variables (.env)")
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    del region, subscription_key
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
    return speech_config


def read_to_list(path):
    with open(path, "r") as f:
        content = f.read()
        words = list(filter(None, content.split("\n")))
        return words
    

def train_test_split(ls: list, test_size: float = 0.2):
    split_idx = int(len(ls) * test_size)
    test = ls[:split_idx]
    train = ls[split_idx:]
    return (train, test)


def generate_audios(words, voices, speech_config, output_folder, isTest=True, silence_len=1000, silence_padding=500, silence_thresh=-40):

    os.makedirs(output_folder, exist_ok=isTest)
    
    if isTest:
        def tqdm(x):
            return x
        rates, pitches, styles, roles = [0], [0], [None], [None]
    else:
        from tqdm import tqdm
        rates, pitches, styles, roles = RATES, PITCHES, STYLES, ROLES
        if rates == [] or pitches == [] or styles == [] or roles == []:
            raise ValueError

    for voice in tqdm(voices):
        count = 0
        for word in words:
            for rate in rates:
                for pitch in pitches:
                    for style in styles:
                        for role in roles:
                            locale = voice[:5]   # e.g., "es-US"
                            people = voice[6:]   # e.g., "PalomaNeural"

                            ssml = create_ssml(
                                locale, voice, word, rate, pitch, style, role
                            )

                            wav_name = f"{people}{count}_{word}.wav"
                            out_path = os.path.join(output_folder, wav_name)

                            stream = speechsdk.audio.PullAudioOutputStream()
                            audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
                            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

                            result = synthesizer.speak_ssml_async(ssml).get()
                            audio_data = result.audio_data

                            # trim end silence
                            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
                            trimmed = effects.strip_silence(audio, silence_thresh=silence_thresh, silence_len=silence_len, padding=silence_padding)
                            trimmed.export(out_path, format="wav")

                            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                                if isTest:
                                    print(f"✅ Synthesized {word} -> {out_path}")
                            elif result.reason == speechsdk.ResultReason.Canceled:
                                cancellation = result.cancellation_details
                                print(f"❌ CANCELED: {cancellation.reason}")
                                if cancellation.reason == speechsdk.CancellationReason.Error:
                                    print(f"ErrorCode: {cancellation.error_code}")
                                    print(f"ErrorDetails: {cancellation.error_details}")

                            count += 1


if __name__ == "__main__":
    main()