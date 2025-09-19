# WakeWord audio gen

**Objective:** Generate audios to train a wake word spotting model.

**Problem:** Synthetic speakers from Text-To-Speech (TTS) models usually generate the same audio output on the same input string, this is not ideal for training a Wake-Word-Spotting (WWS) model. 

**Solution:** This can be overcome by manually forcing phonetic variations on the input strings. 

To create phonetical variations manually, what works best is modifying letters (i -> y) or adding dashes (-) commas (,) and spaces ( ).

Example: "ayud", "a yud", "aiut", "haiut".

*Note: Long and phonetically complex words are recommended: better "hola pepito" than "teo".*

## Usage

Inside ``txt/``, create a folder named ``<your_wakeword>/`` with two files: ``positive_<your_wakeword>.txt`` & ``negative_<your_wakeword>.txt``
Then add any words you want for positive or negative (adversarial) tts generation

Run a test generation with a single voice with
```` bash
uv run python main.py --wakeword hola-pepito --test
````

When you find optimal results, run with all voices. __Note__: *this will consume from your azure subscription, so be careful*
```` bash
uv run python main.py --wakeword hola-pepito
````

## Setup environment

````bash
uv sync
````

Create .env file
````
AZURE_SPEECH_KEY=<key>
AZURE_SPEECH_REGION=<region>
````

## Reference

- TTS docs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
