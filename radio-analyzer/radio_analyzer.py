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
    for sting in eng:
        eng_single = eng_single + " " + str(sting)
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


def finetune_bert(path_data, save_path):

    # Labels : labels = ["O", "B-NAME", "I-NAME", "B-ORT", "I-ORT", "B-DIENSTGRAD", "I-DIENSTGRAD", "B-FAHRZEUG",
    # "I-FAHRZEUG", "B-WAFFE", "I-WAFFE", "B-TOTER", "I-TOTER", "B-CODENAME", "I-CODENAME"]

    num_labels = 15
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = []
    labels = []

    for root, dirs, files in os.walk(path_data):
        for dir in dirs:
            data = open(os.path.join(root, dir, "translation_german"), encoding="utf-8")
            labels.append(open(os.path.join(root, dir, "labels")))
            tmp = ""
            for string in data:
                tmp = tmp + " " + str(string)
            texts.append(tmp)

    tokenized_texts = [tokenizer.tokenize(text) for text in texts]
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]
    attention_masks = [[1] * len(input_id) for input_id in input_ids]

    # convert to tensors
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    # create dataloader
    batch_size = 4
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Save model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)