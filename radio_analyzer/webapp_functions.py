import shutil
import gradio as gr
from . import utils
from . import analyzer
import json
import os
import ast
from geopy.geocoders import Nominatim
import plotly.graph_objects as go


def parse_inputs(cleanup, path, custom, reduce_noise, to_txt, t_model):
    """
        Converts input strings to appropriate boolean values or strings that the Radio Analyzer can
        interpret and returns them.

        :param t_model: Specifies the translation model to be used.
        :param cleanup: Determines whether temporary data should be cleaned up after processing.
        :param to_txt: If set to True, both the transcript and translation will be saved as additional text files.
        :param reduce_noise: If set to True, the Radio Analyzer will attempt to reduce noise in the audio file.
        :param custom: Provides a custom name for the folder where audio chunks, transcriptions,
                and translations are stored. Defaults to a generated name if not specified.
        :param path: Specifies the root directory where analysis folders will be created.
                Can be overridden with a custom path.
        :return: Returns converted values.
        """
    # Transfer Clean up string into boolean

    if cleanup == 'No':
        cleanup = False
    else:
        cleanup = True

    # Clean empty inputs

    if path == '':
        path = os.path.join('~', '.radio_analyzer')

    if custom == '':
        custom = None

    if reduce_noise == 'No':
        reduce_noise = False
    else:
        reduce_noise = True

    if to_txt == 'No':
        to_txt = False
    else:
        to_txt = True
        cleanup = False

    if t_model == 'Helsinki':
        t_model = 'Helsinki-NLP/opus-mt-ru-en'
    elif t_model == 'Facebook':
        t_model = 'facebook/wmt19-ru-en'
    else:
        t_model = 'Whisper'

    return cleanup, path, custom, reduce_noise, to_txt, t_model


def run_analyses(obj, w_model, custom, path, reduce_noise, to_txt, cleanup, t_model):
    """
    Funktion starts analyses process and returns values to gradio webapp.

    :param t_model: Specifies the translation model to be used.
    :param cleanup: Indicates whether temporary data should be cleaned up post-processing.
    :param to_txt: If set to True, the transcript and translation will be additionally saved as text files.
    :param reduce_noise: If set to True, the Radio Analyzer will attempt to reduce noise from the audio file.
    :param w_model: Specifies the version or size of the Whisper model to be used for analysis.
    :param obj: A _TemporaryFileWrapper object created by Gradio, containing the path to the audio file.
    :param custom: Designates a custom name for the folder where audio chunks, transcriptions, and translations are saved.
        If not specified, a default name will be generated.
    :param path: Defines the root directory where analysis folders are created. This can be replaced with a custom path if needed.
    :return: Outputs data in a format suitable for display within the Gradio application.
    """

    if not isinstance(obj, str):
        path_to_audio = obj.name
    else:
        path_to_audio = os.path.expanduser(os.path.join(path, 'analysis_data', obj))

    cleanup, path, custom, reduce_noise, to_txt, t_model = parse_inputs(cleanup, path, custom,
                                                                        reduce_noise, to_txt, t_model)

    # Get file format

    path_to_audio = path_to_audio.replace('\"', '')

    _, file_format = os.path.splitext(path_to_audio)
    file_format = file_format[1:]

    # If json, load file. If mp3,wav, analyse file. Else return error

    if file_format == 'mp4':

        to_remove = path_to_audio
        path_to_audio = utils.extract_audio(path_to_audio)
        file_format = 'mp3'

    if file_format == 'json':
        with open(path_to_audio, 'r') as jfile:
            data = json.load(jfile)

    elif file_format == 'mp3' or file_format == 'wav':
        data = analyzer.analyze_module(path_to_audio,
                                       whisper_model=w_model,
                                       clean_up=cleanup,
                                       custom_name=custom,
                                       base_path=path,
                                       reduce_noise=reduce_noise,
                                       to_txt=to_txt,
                                       translation_model=t_model
                                       )

    else:
        print('Wrong file format')
        raise gr.Error('Wrong File format')

    data_sentiment = data['sentiment']
    ner = ast.literal_eval(data['ner'])

    for entity in ner:
        entity['entity'] = entity.pop('entity_group')

    labels_tactical = {lab: conf for lab, conf in data['labels_tactical']}
    labels_legal = {lab: conf for lab, conf in data['labels_legal']}
    mood_data = data['label_mood']['label']
    english = data['english']
    original = data['original']
    name = data['file_name']
    audio_path = data['path']
    whisper_model = data['model']
    ctime = data['time_of_analysis']

    # Remove from Gradio created tmp-files to clear storage

    if not isinstance(obj, str):
        shutil.rmtree(os.path.dirname(path_to_audio))
        shutil.rmtree(os.path.dirname(obj.orig_name))

    locations = [x for x in ner if x.get('entity') == 'LOC']

    return ({'text': english,
            'entities': ner}, data_sentiment, mood_data, labels_tactical, labels_legal,
            original, name, audio_path, whisper_model, ctime, data, create_map(locations))


def save_conf(cleanup, reduce_noise, to_txt, w_model, t_model, path):
    """
    Saves configuration settings to config.json.

    :param t_model: Specifies the translation model to be used.
    :param cleanup: Indicates whether temporary data should be cleaned up after processing.
    :param to_txt: If set to True, both the transcript and translation will be saved as separate text files.
    :param reduce_noise: If set to True, the Radio Analyzer will attempt to reduce noise from the audio input.
    :param w_model: Specifies the version or size of the Whisper model to be used.
        If not specified, a default version will be used.
    :param path: Designates the root directory where analysis folders will be created. This can be replaced with a custom path if needed.

    """

    if path == '':
        path = os.path.join('~', '.radio_analyzer')

    config = {
                "cleanup": cleanup,
                "reduce_noise": reduce_noise,
                "to_txt": to_txt,
                "w_model": w_model,
                "t_model": t_model,
                "base_directory": path
            }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'w') as jfile:
        json.dump(config, jfile, indent=2)

    gr.Info(message='Your configuration has been saved')


def bulk_analysis(obj_list, w_model, custom, path, reduce_noise, to_txt, cleanup, t_model,
                  progress=gr.Progress(track_tqdm=True)):
    """
    Executes a series of audio analyses consecutively.

    :param progress: Gradio progress object to track the analysis progress.
    :param t_model: Specifies the translation model to be used.
    :param cleanup: Indicates whether temporary data should be cleaned up post-analysis.
    :param to_txt: If set to True, both the transcript and translation will be saved as separate text files.
    :param reduce_noise: If set to True, the Radio Analyzer will attempt to reduce noise in the audio input.
    :param w_model: Specifies the version or size of the Whisper model to be used.
    :param obj_list: List of _TemporaryFileWrapper objects created by Gradio, containing paths to the audio files.
    :param custom: Custom name for the directory where audio chunks, transcriptions, and translations will be saved.
        If not specified, a default directory name will be used.
    :param path: Specifies the root directory where analysis directories are created. This can be replaced with a custom path if desired.
    :return: Returns a 'Done' message followed by any warnings encountered during the analysis.
    """

    warnings = ''

    cleanup, path, custom, reduce_noise, to_txt, t_model = parse_inputs(cleanup, path, custom,
                                                                        reduce_noise, to_txt, t_model)

    for obj in progress.tqdm(obj_list):
        progress(0, desc='Starting...')
        path_to_audio = obj.name

        # Get file format

        path_to_audio = path_to_audio.replace('\"', '')

        _, file_format = os.path.splitext(path_to_audio)
        file_format = file_format[1:]

        if file_format == 'mp3' or file_format == 'wav':
            try:
                analyzer.analyze_module(path_to_audio, whisper_model=w_model, clean_up=cleanup,
                                        custom_name=custom, base_path=path, reduce_noise=reduce_noise,
                                        to_txt=to_txt, translation_model=t_model)
                shutil.rmtree(os.path.dirname(path_to_audio))
                shutil.rmtree(os.path.dirname(obj.orig_name))
            except:
                error = f'An Error occurred with file {file_format}!'
                warnings += error + '\n'
                gr.Warning(error)

    return 'Done! \n Warnings: ' + warnings


def get_json_list():
    """
    Reloads the values for a Gradio dropdown interface element.

    :return: Returns an instance of the gradio Dropdown.Update object to refresh the dropdown values.
    """

    with open('config.json', 'r') as jfile:
        config = json.load(jfile)
    json_path = os.path.expanduser(os.path.join(config['base_directory'], 'analysis_data'))
    file_list = [file for file in os.listdir(json_path) if os.path.isfile(os.path.join(json_path, file))]
    return gr.Dropdown.update(choices=file_list)


def create_map(loc_entries):

    if loc_entries:
        geolocator = Nominatim(user_agent='geoapi')
        locations = []
        lat = []
        long = []
        for entry in loc_entries:
            tmp = entry['word']
            location = geolocator.geocode(tmp)
            if location:
                locations.append(tmp)
                lat.append(location.latitude)
                long.append(location.longitude)
        if locations:
            fig = go.Figure(go.Scattermapbox(
                customdata=locations,
                lat=lat,
                lon=long,
                mode='markers',
                marker=go.scattermapbox.Marker(size=10),
                hoverinfo='text',
                text=locations
            ))

            fig.update_layout(
                mapbox_style='open-street-map',
                hovermode='closest',
                mapbox=dict(
                    bearing=0,
                    center=go.layout.mapbox.Center(lat=(sum(lat)/len(lat)),  lon=(sum(long)/len(long))),
                    pitch=0,
                    zoom=5
                )
            )
            return fig
        return None
