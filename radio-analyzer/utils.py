import whisper
import os
from pydub import AudioSegment
import ntpath
import shutil
import noisereduce as nr
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import torch


def reduce_noise(path, file):
    rate, data = wavfile.read(os.path.join(path, file))
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(os.path.join(path, "reduced_noise_" + file), rate, reduced_noise)


def splitter(recording: AudioSegment, section_start, section_finish, section_counter, folder_path, file_format):
    working_part = recording[section_start:section_finish]
    working_part.export(os.path.join(folder_path, "{:06d}".format(section_counter) + ".mp3"), format=file_format)


def split_audio(path, safe_path):
    file_format = path.split('.', 1)[1]
    recording = AudioSegment.from_file(path, format=path.split('.', 1)[1])

    # Create Folder for Output
    file_number = ntpath.basename(path).split(".", 1)[0]
    tmp = path.split("\\")
    filename = tmp[len(tmp) - 3] + "-" + tmp[len(tmp) - 2]
    folder_path = os.path.join(safe_path, filename + "-" + file_number)

    # Remove preexisting old files
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    full_length = len(recording)
    section_start, section_finish, section_counter = 0, 30000, 1
    if full_length < section_finish:
        print("File is to short and needs no splitting")
        recording.export(os.path.join(folder_path, "{:06d}".format(section_counter) + ".mp3"), format=file_format)
    else:
        while section_finish <= full_length:
            splitter(recording, section_start, section_finish, section_counter, folder_path, file_format)
            section_start = section_finish - 2000
            section_finish = section_finish + 28000
            section_counter += 1
        section_finish = full_length
        splitter(recording, section_start, section_finish, section_counter, folder_path, file_format)
    return folder_path


def transcribe_to_txt(chunk_path, whisper_model="large", german_translation_model="helsinki"):

    """
        chunk_path: path to the audio chunks. Chunks cannot be longer than 30 seconds
        model: Size of the whisper model you want to use.
                Available sizes are: tiny, base, small, medium, and large.
                For references, visit: https://github.com/openai/whisper
        german_translation_model: model for the translation english-german
                Available options: helsinki, nllb
    """


    # german-english-model

    if german_translation_model == "helsinki":
        tokenizer_helsinki = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        model_helsinki = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    elif german_translation_model == "nllb":
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', tokenizer='moses', bpe='subword_nmt')
        en2de.eval()
        en2de.cuda()

    # Whisper

    whisper_model = whisper_model

    filename = chunk_path.rsplit("\\", 1)[1]
    print("File: " + filename)

    text_original = []
    text_english = []
    text_german = []

    text_original.append("Model: whisper-large-v2 Task: transcribe\n".encode('utf-8'))
    text_original.append(("###START OF ORIGINAL TRANSCRIPTION FROM FILE " + str(filename) + "###").encode('utf-8'))
    text_english.append("Model: whisper-large-v2 Task: translate original-english\n")
    text_english.append("###START OF ENGLISH TRANSLATION FROM FILE " + str(filename) + "###")
    text_german.append("Model: Helsinki-nlp Task: translate english-german\n")
    text_german.append("###START OF GERMAN TRANSLATION FROM FILE " + str(filename) + "###")

    model = whisper.load_model(whisper_model)

    time_start = 0
    time_end = 30

    for file in os.listdir(chunk_path):
        print("Working on part: " + file)

        # Transcription + English translation

        original_transcript = model.transcribe(os.path.join(chunk_path, file), task="transcribe")["text"].encode('utf-8')
        text_english = model.transcribe(os.path.join(chunk_path, file), task="translate")["text"]

        text_original.append(("\n######## START OF " + str(os.path.basename(file)).upper()
                              + " (" + str(datetime.timedelta(seconds=time_start)) + " - "
                              + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n").encode('utf-8'))
        text_original.append(original_transcript.strip())

        text_english.append("\n######## START OF " + str(os.path.basename(file)).upper()
                            + " (" + str(datetime.timedelta(seconds=time_start))
                            + " - " + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n")
        text_english.append(text_english.strip())

        # Helsinki

        text_german.append("\n######## START OF " + str(os.path.basename(file)).upper()
                           + " (" + str(datetime.timedelta(seconds=time_start)) + " - "
                           + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n")

        if german_translation_model == "helsinki":
            input_ids_helsinki = tokenizer_helsinki.encode(text_english, return_tensors="pt")
            outputs_helsinki = model_helsinki.generate(input_ids_helsinki)
            decoded_helsinki = tokenizer_helsinki.decode(outputs_helsinki[0], skip_special_tokens=True)

            text_german.append(decoded_helsinki)

        elif german_translation_model == "nllb":
            text_german.append(en2de.translate(text_english))

        time_start = time_end - 2
        time_end = time_start + 30

    # Encode to UTF-8 to avoid encoding errors caused by cyrillic characters

    eng_out = [x.encode('utf-8') for x in text_english]
    deu_out = [x.encode('utf-8') for x in text_german]

    # Create output txt-files

    with open(os.path.join(chunk_path, "translation_english.txt"), 'wb') as f:
        for entry in eng_out:
            f.write(entry)
    f.close()
    with open(os.path.join(chunk_path, "transcript_original.txt"), 'wb') as f:
        for entry in text_original:
            f.write(entry)
    f.close()
    with open(os.path.join(chunk_path, "translation_deutsch.txt"), 'wb') as f:
        for entry in deu_out:
            f.write(entry)
    f.close()


def transcribe(chunk_path, whisper_model="large", german_translation_model="helsinki"):
    """
        chunk_path: path to the audio chunks. Chunks cannot be longer than 30 seconds
        model: Size of the whisper model you want to use.
                Available sizes are: tiny, base, small, medium, and large.
                For references, visit: https://github.com/openai/whisper
        german_translation_model: model for the translation english-german
                Available options: helsinki, nllb
    """

    # german-english-model

    if german_translation_model == "helsinki":
        tokenizer_helsinki = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        model_helsinki = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    elif german_translation_model == "nllb":
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', tokenizer='moses', bpe='subword_nmt')
        en2de.eval()
        en2de.cuda()

    # Whisper

    whisper_model = whisper_model

    filename = chunk_path.rsplit("\\", 1)[1]
    print("File: " + filename)

    text_original = []
    text_english = []
    text_german = []

    model = whisper.load_model(whisper_model)

    time_start = 0
    time_end = 30

    for file in os.listdir(chunk_path):
        print("Working on part: " + file)

        # Transcription + English translation

        original_transcript = model.transcribe(os.path.join(chunk_path, file), task="transcribe")["text"].encode(
            'utf-8')
        text_english = model.transcribe(os.path.join(chunk_path, file), task="translate")["text"]

        text_original.append(original_transcript.strip())

        text_english.append(text_english.strip())

        # Helsinki

        if german_translation_model == "helsinki":
            input_ids_helsinki = tokenizer_helsinki.encode(text_english, return_tensors="pt")
            outputs_helsinki = model_helsinki.generate(input_ids_helsinki)
            decoded_helsinki = tokenizer_helsinki.decode(outputs_helsinki[0], skip_special_tokens=True)

            text_german.append(decoded_helsinki)

        elif german_translation_model == "nllb":
            text_german.append(en2de.translate(text_english))

        time_start = time_end - 2
        time_end = time_start + 30

    # Encode to UTF-8 to avoid encoding errors caused by cyrillic characters

    eng_out = [x.encode('utf-8') for x in text_english]
    ger_out = [x.encode('utf-8') for x in text_german]

    return text_original, eng_out, ger_out
