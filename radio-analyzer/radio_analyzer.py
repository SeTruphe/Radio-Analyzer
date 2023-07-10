import datetime
import os.path
import shutil
import datasets
import numpy as np
from transformers import BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import pipeline, AutoTokenizer

import utils


def radio_analyzer(audio_path, custom_name=None, cleanup=False, base_path=os.path.join("~", ".radio_analyzer")):
    """
    :param audio_path: path to the audiofile you want to analyse
    :param custom_name: The app creates a folder for the audio chunks as well as transcription and translation text files.
            The folder name is generated automatically. You can alter the folder name here.
    :param cleanup: If set to true, generated folder for the project is deleted after the analysis.
            Set to true to safe space. Default is false.
    :param base_path: The folders for the analysis are generated in the base path of the user.
            You can define a different path here.
    :return: None.
    """

    # Generate folder for current file
    if custom_name:
        path = os.path.expanduser(os.path.join(base_path, custom_name))
    else:
        ct = datetime.datetime.now()
        path = os.path.expanduser(os.path.join(base_path, str(ct).replace(":", "-").replace(" ", "-")
                                               .replace(".", "-")))

    os.makedirs(path)

    # Process audio file and get transcription and translations

    chunk_path = utils.split_audio(audio_path, path, custom_name)
    org, eng, ger = utils.transcribe(chunk_path, internal_mode=True, to_txt=True)

    # Analyse

    ner_model = 'dslim/bert-base-NER'
    tokenizer = BertTokenizerFast.from_pretrained(ner_model)
    model = AutoModelForTokenClassification.from_pretrained(ner_model)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_results = []
    for string in eng:
        ner_results.append(nlp(str(string)))
    print(ner_results)

    if cleanup:
        shutil.rmtree(path)

