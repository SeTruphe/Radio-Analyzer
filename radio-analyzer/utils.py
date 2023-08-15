import whisper
import os
from pydub import AudioSegment
import ntpath
import shutil
import noisereduce as nr
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel
import datetime


def noisereduce(file):

    """
    :param file: Path to the audio file for noise reduction.
    :return: Path to the processed audio file with reduced noise.
    """

    folder_path = os.path.dirname(file)
    file_name, _ = os.path.splitext(os.path.basename(file))
    file_format = os.path.splitext(file)[1].lstrip('.')

    # if file is mp3, convert to wav and create a tmp file

    if file_format == "mp3":
        audio = AudioSegment.from_mp3(file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        tmp = os.path.join(folder_path, "tmp.wav")
        audio.export(tmp, format="wav")
        file = tmp

    # do noisereduce on wav

    output = os.path.join(folder_path, file_name + "_reduced.wav")
    rate, data = wavfile.read(file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(output, rate, reduced_noise)

    # if tmp file was created, remove

    if file_format == "mp3":
        os.remove(tmp)

    return output


def splitter(recording: AudioSegment, section_start, section_finish, section_counter, folder_path, file_format):
    """
    :param recording: Audio file of type 'AudioSegment' to be split.
    :param section_start: Start time of the segment in seconds.
    :param section_finish: End time of the segment in seconds.
    :param section_counter: Continuous number for the segment.
    :param folder_path: Path where the segment will be saved.
    :param file_format: Desired file format for the segment.
    :return: None

    This function splits an audio file into 30-second chunks.
    """
    working_part = recording[section_start:section_finish]
    working_part.export(os.path.join(folder_path, '{:06d}'.format(section_counter) + '.mp3'), format=file_format)


def split_audio(audio_path, save_path, opt_folder_name=None):
    """
    :param audio_path: Path to the audio file intended for chunking.
    :param save_path: Directory where the audio chunks will be stored.
    :param opt_folder_name: Optional parameter to customize the initial part of the save folder's name.
        If left unspecified, the folder name will be derived from the audio file's directory name.
        The latter part of the folder name will always be the name of the audio file.
    :return: Returns the full directory path containing the audio chunks.
    """

    file_format = audio_path.split('.', 1)[1]
    recording = AudioSegment.from_file(audio_path, format=audio_path.split('.', 1)[1])

    # Create Folder for Output

    name_arg_2 = ntpath.basename(audio_path).split(".", 1)[0]
    if opt_folder_name:
        name_arg_1 = opt_folder_name
    else:
        tmp = audio_path.split('\\')
        name_arg_1 = tmp[len(tmp) - 3] + '-' + tmp[len(tmp) - 2]
    folder_path = os.path.join(save_path, name_arg_1 + '-' + name_arg_2)

    # Remove preexisting old files

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    full_length = len(recording)
    section_start, section_finish, section_counter = 0, 30000, 1
    if full_length < section_finish:
        print('File is to short and needs no splitting')
        recording.export(os.path.join(folder_path, '{:06d}'.format(section_counter) + '.mp3'), format=file_format)
    else:
        while section_finish <= full_length:
            splitter(recording, section_start, section_finish, section_counter, folder_path, file_format)
            section_start = section_finish - 2000
            section_finish = section_finish + 28000
            section_counter += 1
        section_finish = full_length
        splitter(recording, section_start, section_finish, section_counter, folder_path, file_format)
    return folder_path


def transcribe_parts(chunk_path, whisper_model='large-v2', internal_mode=False, to_txt=False):
    """
    :param chunk_path: Path to the audio chunks. Each chunk must not exceed 30 seconds in duration.
    :param whisper_model: Specifies the size of the Whisper model to be used.
        Available options include: tiny, base, small, medium, large, and large-v2.
        For more details, refer to: https://github.com/openai/whisper
    :param internal_mode: If set to true, explanatory and delimiter strings are omitted from the transcript/translation
        lists, facilitating further internal processing.
    :param to_txt: If true, both the transcript and translations are saved as .txt files within the chunk directory.
    :return: Returns lists containing the transcriptions and translations.
    """

    # german-english-model

    tokenizer_helsinki = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
    model_helsinki = AutoModelForSeq2SeqLM.from_pretrained('Helsinki-NLP/opus-mt-en-de')

    # Whisper

    whisper_model = whisper_model

    filename = chunk_path.rsplit('\\', 1)[1]
    print('File: ' + filename)

    text_original = []
    text_english = []
    text_german = []

    if not internal_mode:
        text_original.append(('Model: whisper-' + whisper_model + ' Task: transcribe\n').encode('utf-8'))
        text_original.append(('###START OF ORIGINAL TRANSCRIPTION FROM FILE ' + str(filename) + '###').encode('utf-8'))
        text_english.append('Model: whisper-large-v2 Task: translate original-english\n')
        text_english.append('###START OF ENGLISH TRANSLATION FROM FILE ' + str(filename) + '###')
        text_german.append('Model: Helsinki-nlp Task: translate english-german\n')
        text_german.append('###START OF GERMAN TRANSLATION FROM FILE ' + str(filename) + '###')

    model = whisper.load_model(whisper_model)

    time_start = 0
    time_end = 30

    for file in os.listdir(chunk_path):
        print('Working on part: ' + file)

        # Transcription + English translation

        original_transcript = model.transcribe(os.path.join(chunk_path, file), task='transcribe')['text'].encode(
            'utf-8')
        translation_english = model.transcribe(os.path.join(chunk_path, file), task='translate')['text']

        if not internal_mode:
            text_original.append(('\n######## START OF ' + str(os.path.basename(file)).upper()
                                  + ' (' + str(datetime.timedelta(seconds=time_start)) + ' - '
                                  + str(datetime.timedelta(seconds=time_end)) + 's) ' + '########\n').encode('utf-8'))
            text_english.append('\n######## START OF ' + str(os.path.basename(file)).upper()
                                + ' (' + str(datetime.timedelta(seconds=time_start))
                                + ' - ' + str(datetime.timedelta(seconds=time_end)) + 's) ' + '########\n')
            text_german.append('\n######## START OF ' + str(os.path.basename(file)).upper()
                               + ' (' + str(datetime.timedelta(seconds=time_start)) + ' - '
                               + str(datetime.timedelta(seconds=time_end)) + 's) ' + '########\n')

        text_original.append(original_transcript.strip())
        text_english.append(translation_english.strip())

        input_ids_helsinki = tokenizer_helsinki.encode(translation_english, return_tensors='pt')
        outputs_helsinki = model_helsinki.generate(input_ids_helsinki)
        decoded_helsinki = tokenizer_helsinki.decode(outputs_helsinki[0], skip_special_tokens=True)
        text_german.append(decoded_helsinki)

        time_start = time_end - 2
        time_end = time_start + 30

    # Encode to UTF-8 to avoid encoding errors caused by cyrillic characters

    eng_out = [x.encode('utf-8') for x in text_english]
    deu_out = [x.encode('utf-8') for x in text_german]

    # Create output txt-files
    if to_txt:
        with open(os.path.join(chunk_path, 'translation_english.txt'), 'wb') as f:
            for entry in eng_out:
                f.write(entry)
        f.close()
        with open(os.path.join(chunk_path, 'transcript_original.txt'), 'wb') as f:
            for entry in text_original:
                f.write(entry)
        f.close()
        with open(os.path.join(chunk_path, 'translation_german.txt'), 'wb') as f:
            for entry in deu_out:
                f.write(entry)
        f.close()

    return text_original, eng_out, deu_out


def transcribe(file, whisper_model='large-v2', to_txt=False):
    """
    :param file: Path to the audio file intended for transcription and translation.
    :param whisper_model: Specifies the size of the Whisper model to be used.
        Options include: tiny, base, small, medium, large, and large-v2.
        For more details, refer to: https://github.com/openai/whisper
    :param to_txt: If set to true, both the transcript and translations are saved as .txt files in the file's directory.
    :return: Returns lists of transcriptions and translations.
    """
    file_name = os.path.splitext(file)[0]
    folder_path = os.path.dirname(file)

    # Whisper

    whisper_model = whisper_model

    model = whisper.load_model(whisper_model)

    # Transcription + English translation

    text_original = model.transcribe(file, task='transcribe')['text'].strip().encode('utf-8')
    text_english = model.transcribe(file, task='translate')['text']
    text_english = text_english.strip()

    # Encode to UTF-8 to avoid encoding errors caused by cyrillic characters

    text_english = text_english.encode('utf-8')

    # Get filename and folder path, create new folder for results

    if os.path.exists(os.path.join(folder_path, file_name)):
        shutil.rmtree(os.path.join(folder_path, file_name))
    os.makedirs(os.path.join(folder_path, file_name))

    # Create output txt-files

    if to_txt:
        with open(os.path.join(folder_path, file_name, 'translation_english.txt'), 'wb') as f:
            f.write(text_english)
        f.close()
        with open(os.path.join(folder_path, file_name, 'transcript_original.txt'), 'wb') as f:
            f.write(text_original)
        f.close()

    return text_original.decode('utf-8'), text_english.decode('utf-8')

