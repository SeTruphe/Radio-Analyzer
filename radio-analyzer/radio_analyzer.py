import datetime
import os.path
import shutil
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from collections import Counter
import utils
import json
import re


def radio_analyzer(audio_path, custom_name=None, clean_up=False, base_path=os.path.join('~', '.radio_analyzer'),
                   whisper_model='large-v2', to_txt=False, reduce_noise=False):
    """
    :param audio_path: path to the audiofile you want to analyse
    :param custom_name: The app creates a folder for the audio chunks as well as transcription and translation text
            files. The folder name is generated automatically. You can alter the folder name here.
    :param clean_up: If set to true, generated folder for the project is deleted after the analysis.
            Set to true to safe space. Default is false.
    :param base_path: The folders for the analysis are generated in the base path of the user.
            You can define a different path here.
    :param to_txt: if true, the transcript and translations will be saved into txt files in the file of the Audio file.
    :param whisper_model: whisper_model: Size of the whisper model you want to use.
            Available sizes are: tiny, base, small, medium, large and large-v2.
            For references, visit: https://github.com/openai/whisper
    :param reduce_noise: if set to true, noisereduce will try to reduce noise on the Audio file before analysis.
            Default is false
    :return: returns the data dict to display the results in the Webapp.
    """

    # Generate folder for current file

    ct = datetime.datetime.now()
    file_name = os.path.splitext(os.path.basename(audio_path))[0]

    # saving audio_path in new var which can be altered by noisereduce, so "_reduced" ending does not appear later

    working_path = audio_path

    if reduce_noise:
        working_path = utils.noisereduce(audio_path)
    if custom_name:
        file_name = custom_name
    else:
        file_name = file_name + '_' + str(ct).replace(':', '-').replace(' ', '-').replace('.', '-')

    path = os.path.expanduser(os.path.join(base_path, file_name))

    os.makedirs(path)

    # Process audio file and get transcription and translations

    original, english = utils.transcribe(working_path, whisper_model=whisper_model, to_txt=to_txt)

    # if noisereduce and clean_up, remove file after processing to reduce space

    if reduce_noise and clean_up:
        os.remove(working_path)

    # NER-Analysis

    ner_model = 'dslim/bert-base-NER'
    tokenizer = BertTokenizer.from_pretrained(ner_model)
    model = BertForTokenClassification.from_pretrained(ner_model)
    nlp = pipeline('ner', model=model.to('cpu'), tokenizer=tokenizer, aggregation_strategy='simple')

    ner_results = nlp(english)

    # Sentiment Analysis

    # Load model and Chunk text into parts of model max length

    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    max_seq_length = sentiment_analyzer.tokenizer.model_max_length

    segments = [english[i:i + max_seq_length] for i in range(0, len(english), max_seq_length)]

    # Analyse Chunks

    aggregate_result = {'label': [], 'score': []}
    for segment in segments:
        result = sentiment_analyzer(segment)
        for res_dict in result:
            aggregate_result['label'].append(res_dict['label'])
            aggregate_result['score'].append(res_dict['score'])

    # Majority Voting for Sentiment

    majority_label = Counter(aggregate_result['label']).most_common(1)[0][0]

    print(majority_label)

    # save to path

    save_path = os.path.join(os.path.expanduser(base_path), 'analysis_data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Zero Shot Text classification

    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    classes_tactical = ['Unclassified', 'Logistic and Supplies', 'Casualty Report', 'Reconnaissance activities',
                        'Troop movement', 'Military strategy discussion', 'Plans for future operations']
    classes_legal = ['Unclassified', 'Looting', 'Crimes', 'Rape', 'Violation of international law', 'Pillage']

    classes_mood = ['Aggressiv', 'Defensive', 'Concerned', 'Optimistic']

    classifier_result_tactical = classifier(english, classes_tactical)
    class_results_legal = classifier(english, classes_legal)
    class_results_mood = classifier(english, classes_mood)

    labels_tactical = classifier_result_tactical['labels']
    scores_tactical = classifier_result_tactical['scores']

    labels_legal = class_results_legal['labels']
    scores_legal = class_results_legal['scores']

    labels_mood = class_results_mood['labels']
    scores_mood = class_results_mood['scores']

    for i in range(len(labels_tactical)):
        print(f'(Label: {labels_tactical[i]}, Score: {round(scores_tactical[i] * 100, 1)}%)')

    for i in range(len(labels_legal)):
        print(f'(Label: {labels_legal[i]}, Score: {round(scores_legal[i] * 100, 1)}%)')

    for i in range(len(labels_mood)):
        print(f'(Label: {labels_mood[i]}, Score: {round(scores_mood[i] * 100, 1)}%)')

    pairs_tactical = list(zip(labels_tactical, scores_tactical))
    pairs_legal = list(zip(labels_legal, scores_legal))

    # This is a workaround as the model does not provide start and end positions right now

    offset = 0
    to_remove = []

    for entity_group in ner_results:
        word = entity_group['word']
        start_pos = english.find(word, offset)
        if start_pos == -1:
            match = re.search(r'\b{}\b'.format(word).replace("#", ""), english[offset:])
            if match:
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                entity_group['start'] = start_pos
                entity_group['end'] = end_pos
            else:
                to_remove.append(entity_group)
        else:
            # Update the entity group with the start and end positions
            end_pos = start_pos + len(word)
            entity_group['start'] = start_pos
            entity_group['end'] = end_pos

        # Update the offset to search from the end of the last found entity

        offset = end_pos

    # Remove remaining entities with None values as start or endposition to avoid Errors in visualiser

    for entity in to_remove:
        ner_results.remove(entity)

    # Create json

    data = {'sentiment': str(majority_label),
            'ner': str(ner_results),
            'major_tactical': {'label': labels_tactical[0], 'score': scores_tactical[0]},
            'major_legal': {'label': labels_legal[0], 'score': scores_legal[0]},
            'label_mood': {'label': labels_mood[0], 'confidence': scores_mood[0]},
            'labels_tactical': [pair for pair in pairs_tactical],
            'labels_legal': [pair for pair in pairs_legal],
            'original': original,
            'english': english,
            'file_name': os.path.basename(audio_path),
            'path': audio_path,
            'model': whisper_model,
            'time_of_analysis': str(ct)
            }

    with open(os.path.join(save_path, file_name + '.json'), 'w') as jfile:
        json.dump(data, jfile)

    if clean_up:
        shutil.rmtree(path)
    return data
