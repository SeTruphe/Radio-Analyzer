import datetime
import os.path
import shutil
import datasets
import numpy as np
import tokenizers
from transformers import BertTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import pipeline, AutoTokenizer, BertForTokenClassification
from collections import Counter
import utils
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
import json


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
        file_name = custom_name
    else:
        ct = datetime.datetime.now()
        file_name = str(ct).replace(":", "-").replace(" ", "-").replace(".", "-")

    path = os.path.expanduser(os.path.join(base_path, file_name))

    os.makedirs(path)

    # Process audio file and get transcription and translations

    chunk_path = utils.split_audio(audio_path, path, custom_name)
    org, eng, ger = utils.transcribe(chunk_path, internal_mode=True, to_txt=True)

    # NER-Analysis

    ner_model = 'dslim/bert-base-NER'
    tokenizer = BertTokenizer.from_pretrained(ner_model)
    model = AutoModelForTokenClassification.from_pretrained(ner_model)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ner_results = []
    for string in eng:
        ner_results.append(nlp(str(string)))
    print("NER: ", ner_results)

    # Sentiment Analysis

    # Load model and Chunk text into parts of model max length

    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    max_seq_length = sentiment_analyzer.tokenizer.model_max_length

    eng_single = ""
    for string in eng:
        eng_single = eng_single + " " + str(string)
    segments = [eng_single[i:i + max_seq_length] for i in range(0, len(eng_single), max_seq_length)]

    # Analyse Chunks

    aggregate_result = {"label": [], "score": []}
    for segment in segments:
        result = sentiment_analyzer(segment)
        for res_dict in result:
            aggregate_result["label"].append(res_dict["label"])
            aggregate_result["score"].append(res_dict["score"])

    # Majority Voting for Sentiment

    majority_label = Counter(aggregate_result["label"]).most_common(1)[0][0]

    print(majority_label)

    if cleanup:
        shutil.rmtree(path)

    # save to path
    save_path = os.path.join(base_path, "analysis_data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    # Zero Shot Text classification

    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    classes = ["Looting", "Acts of aggression", "Military strategy discussion", "Weapons usage", "Troop movement", "Plans for future operations", "Reconnaissance activities", "Violation of international law", "War crimes", "unclassified", "Rape"]

    clasifier_result = classifier(eng_single, classes)

    labels = clasifier_result["labels"]
    scores = clasifier_result["scores"]

    for i in range(len(labels)):
        print(f"(Label: {labels[i]}, Score: {round(scores[i]*100, 1)}%)")

    data = {"sentiment": str(majority_label),
            "ner": str(ner_results)}

    with open(os.path.join(save_path, file_name + ".json"), 'w') as jfile:
        json.dump(data, jfile)