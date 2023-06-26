import whisper
import os
from pydub import AudioSegment
import ntpath
import shutil
import noisereduce as nr
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime


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


def transcribe_to_txt(path):
    # Helsinki-nlp
    tokenizer_helsinki = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model_helsinki = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    # Whisper

    whisper_model = "large"

    filename = path.rsplit("\\", 1)[1]
    print("File: " + filename)

    text_original = []
    text_english = []
    text_deutsch_helsinki = []

    text_original.append("Model: whisper-large-v2 Task: transcribe\n".encode('utf-8'))
    text_original.append(("###START OF ORIGINAL TRANSCRIPTION FROM FILE " + str(filename) + "###").encode('utf-8'))
    text_english.append("Model: whisper-large-v2 Task: translate original-english\n")
    text_english.append("###START OF ENGLISH TRANSLATION FROM FILE " + str(filename) + "###")
    text_deutsch_helsinki.append("Model: Helsinki-nlp Task: translate english-german\n")
    text_deutsch_helsinki.append("###START OF GERMAN TRANSLATION FROM FILE " + str(filename) + "###")

    model = whisper.load_model(whisper_model)

    time_start = 0
    time_end = 30

    for file in os.listdir(path):
        print("Working on part: " + file)

        # Transcription + English

        transcript = model.transcribe(os.path.join(path, file), task="transcribe")["text"].encode('utf-8')
        translate = model.transcribe(os.path.join(path, file), task="translate")["text"]

        text_original.append(("\n######## START OF " + str(os.path.basename(file)).upper()
                              + " (" + str(datetime.timedelta(seconds=time_start)) + " - "
                              + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n").encode('utf-8'))
        text_original.append(transcript.strip())

        text_english.append("\n######## START OF " + str(os.path.basename(file)).upper()
                            + " (" + str(datetime.timedelta(seconds=time_start))
                            + " - " + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n")
        text_english.append(translate.strip())

        # Helsinki

        input_ids_helsinki = tokenizer_helsinki.encode(translate, return_tensors="pt")
        outputs_helsinki = model_helsinki.generate(input_ids_helsinki)
        decoded_helsinki = tokenizer_helsinki.decode(outputs_helsinki[0], skip_special_tokens=True)
        text_deutsch_helsinki.append("\n######## START OF " + str(os.path.basename(file)).upper()
                                     + " (" + str(datetime.timedelta(seconds=time_start)) + " - "
                                     + str(datetime.timedelta(seconds=time_end)) + "s) " + "########\n")
        text_deutsch_helsinki.append(decoded_helsinki)

        time_start = time_end - 2
        time_end = time_start + 30

    eng_out = [x.encode('utf-8') for x in text_english]
    deu_out_helsinki = [x.encode('utf-8') for x in text_deutsch_helsinki]

    with open(os.path.join(path, "translation_english.txt"), 'wb') as f:
        for entry in eng_out:
            f.write(entry)
    f.close()
    with open(os.path.join(path, "transcript_original.txt"), 'wb') as f:
        for entry in text_original:
            f.write(entry)
    f.close()
    with open(os.path.join(path, "translation_deutsch.txt"), 'wb') as f:
        for entry in deu_out_helsinki:
            f.write(entry)
    f.close()


def transcribe(chunk_path):

    # Helsinki-nlp
    tokenizer_helsinki = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model_helsinki = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    # Whisper

    whisper_model = "large"

    folder_name = chunk_path.rsplit("\\", 1)[1]
    print("Folder: " + folder_name)

    text_original = []
    text_english = []
    text_deutsch_helsinki = []

    model = whisper.load_model(whisper_model)

    time_start = 0
    time_end = 30

    for file in os.listdir(chunk_path):
        print("Working on part: " + file)

        # Transcription + English

        transcript = model.transcribe(os.path.join(chunk_path, file), task="transcribe")["text"].encode('utf-8')
        translate = model.transcribe(os.path.join(chunk_path, file), task="translate")["text"]

        text_original.append(transcript.strip())

        text_english.append(translate.strip())

        # Helsinki

        input_ids_helsinki = tokenizer_helsinki.encode(translate, return_tensors="pt")
        outputs_helsinki = model_helsinki.generate(input_ids_helsinki)
        decoded_helsinki = tokenizer_helsinki.decode(outputs_helsinki[0], skip_special_tokens=True)
        text_deutsch_helsinki.append(decoded_helsinki)

        time_start = time_end - 2
        time_end = time_start + 30

    eng_out = [x.encode('utf-8') for x in text_english]
    deu_out_helsinki = [x.encode('utf-8') for x in text_deutsch_helsinki]

    return text_original, eng_out, deu_out_helsinki
    return text_original, text_english, text_deutsch_helsinki