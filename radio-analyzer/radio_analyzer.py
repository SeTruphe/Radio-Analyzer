import datetime
import os.path
import utils
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def radio_analyzer(audio_path, custom_name=None):

    # Generate folder for current file

    if custom_name:
        path = os.path.expanduser(os.path.join("~", "radio_analyzer", custom_name))
    else:
        ct = datetime.datetime.now()
        path = os.path.expanduser(os.path.join("~", "radio_analyzer", str(ct)))

    os.makedirs(path)

    # Process audio file and get transcription and translations

    chunk_path = utils.split_audio(audio_path, path, custom_name)
    org, eng, ger = utils.transcribe(chunk_path, internal_mode=True)