import shutil
import gradio as gr
from . import utils
from . import analyzer
import json
import os
import ast
import webbrowser
from geopy.geocoders import Nominatim
import plotly.graph_objects as go


def parse_inputs(clean_up, path, custom, reduce_noise, to_txt, t_model):

    # Transfer Clean up string into boolean

    if clean_up == 'No':
        clean_up = False
    else:
        clean_up = True

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
        clean_up = False

    if t_model == 'Helsinki':
        t_model = 'Helsinki-NLP/opus-mt-ru-en'
    elif t_model == 'Facebook':
        t_model = 'facebook/wmt19-ru-en'
    else:
        t_model = 'Whisper'

    return clean_up, path, custom, reduce_noise, to_txt, t_model


def run_and_display(obj, w_model, custom, path, reduce_noise, to_txt, clean_up, t_model):
    """
    :param obj: _TemporaryFileWrapper-Object which is created by Gradio and hast the path to the Audiofile
    :param custom: Custom name for the folder where audio chunks, transcriptions, and translations are stored.
        If not provided, a default name is generated.
    :param path: Root directory where analysis folders are created. Can be overridden with a custom path.
    :return: Returns data formatted for display in the Gradio app.
    """
    if not isinstance(obj, str):
        path_to_audio = obj.name
    else:
        path_to_audio = os.path.expanduser(os.path.join(path, 'analysis_data', obj))

    clean_up, path, custom, reduce_noise, to_txt, t_model = parse_inputs(clean_up, path, custom,
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
                                       clean_up=clean_up,
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

    locs = [x for x in ner if x.get('entity') == 'LOC']


    return ({'text': english,
            'entities': ner}, data_sentiment, mood_data, labels_tactical, labels_legal,
            original, name, audio_path, whisper_model, ctime, data, create_map(locs))

def save_conf(cleanup, reduce_noise, to_txt, w_model, t_model, path):

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


def bulk_analysis(obj_list, w_model, custom, path, reduce_noise, to_txt, clean_up, t_model,
                  progress=gr.Progress(track_tqdm=True)):

    warnings = ''

    clean_up, path, custom, reduce_noise, to_txt, t_model = parse_inputs(clean_up, path, custom,
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
                analyzer.analyze_module(path_to_audio, whisper_model=w_model, clean_up=clean_up,
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
    with open('config.json', 'r') as jfile:
        config = json.load(jfile)
    json_path = os.path.expanduser(os.path.join(config['base_directory'], 'analysis_data'))
    file_list = [file for file in os.listdir(json_path) if os.path.isfile(os.path.join(json_path, file))]
    return gr.Dropdown.update(choices=file_list)


def create_map(loc_entries):

    if loc_entries:
        geoloc = Nominatim(user_agent='geoapi')
        locs = []
        lat = []
        long = []
        for entry in loc_entries:
            tmp = entry['word']
            location = geoloc.geocode(tmp)
            if location:
                locs.append(tmp)
                lat.append(location.latitude)
                long.append(location.longitude)
        if locs:
            fig = go.Figure(go.Scattermapbox(
                customdata=locs,
                lat=lat,
                lon=long,
                mode='markers',
                marker=go.scattermapbox.Marker(size=10),
                hoverinfo='text',
                text=locs
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


# Create Gradio App

working_dir = os.path.dirname(os.path.abspath(__file__))

with gr.Blocks() as analyzer_webapp:
    gr.Image(value=os.path.join(working_dir, 'media', 'wordmark-bright.svg'), height=100, container=False)

    # Creates tab for path input

    with gr.Tab('Analyze'):

        # Tabs for Input

        with gr.Blocks():
            with gr.Tab('File Selector'):
                obj = gr.File(label='Please input the Audio file you want to Analyze', file_count='single',
                              file_types=['.mp3', '.wav', '.mp4'])
                file_selector_button = gr.Button('Analyze', size='lg')

            with gr.Tab('JSON Selector'):
                with gr.Row():
                    json_obj = gr.Dropdown(choices=['Refresh'], label='Select previous analyse result', scale=3)
                    refresh_button = gr.Button(value='Refresh Dropdown', scale=0)
                    refresh_button.click(get_json_list, outputs=json_obj)
                json_button = gr.Button('Load JSON', size='lg')

        # Tabs for Output

        with gr.Blocks():
            with gr.Tab('Analysis Data'):
                    with gr.Row():
                        file_name = gr.components.Textbox(label='Name of the File')
                        file_path = gr.components.Textbox(label='Path to Audio File')
                    with gr.Row():
                        model = gr.components.Textbox(label='Whisper Model Used')
                        time = gr.components.Textbox(label='Start Time of Analysis')
                    with gr.Row():
                        sentiment = gr.components.Textbox(label='Overall Sentiment')
                        mood = gr.components.Textbox(label='Mood of the Text')
                    with gr.Row():
                        label_tac = gr.Label(label='All Tactical Labels')
                        label_legal = gr.Label(label='All Legal Labels')

                    highlight = gr.HighlightedText(label='NER')

                    loc_map = gr.Plot()

            with gr.Tab('Original Text'):
                org = gr.components.Textbox(label='Original Text', lines=10)

            with gr.Tab('Raw Data'):
                js = gr.JSON(label="Raw JSON data")

    # Tab for Bulk Analysis

    with gr.Tab('Bulk Analysis'):
        gr.Markdown("""
                    In this tab, you can choose a folder containing multiple audio files for batch analysis. <br>
                    All the audio files within the selected folder will be automatically processed.
                    Please note: The folder you select will be displayed as empty! After selecting the folder, all
                    contents will be uploaded regardless.
                    """)
        folder_list = gr.File(label='Select a folder', file_count='directory')
        progress = gr.Textbox(label='Progress')
        bulk_button = gr.Button('Bulk Analyze', size='lg')

    # Tab for advanced parameters

    with gr.Tab('Advanced Settings'):
        with open(os.path.join(working_dir,'config.json'), 'r') as jfile:
            conf = json.load(jfile)
        gr.Markdown("""
        In this tab, you can adjust and input the advanced settings of the app.
        Clean Up: Enable this option to automatically delete the newly created folder in the .radio_analyzer directory
         for the current audio file after the analysis process is complete.<br><br>
        Create .txt Files: Enable this option to save the transcriptions and translations as text files within the
         .radio_analyzer folder designated for the audio file.<br>
        Whisper Model: Use this setting to select the Whisper model you'd like to use for analysis.
         The default model is large-v2.<br>
        Translation Model: Use this setting to change the model which translates the original transcript.
          The default is 'Whisper'.<br>
        Save File Name: Specify a custom name for your save file. This name will also serve as the folder
         name within the .radio_analyzer directory.<br>
        Adjust Base Directory: If you wish to use a different base directory for .radio_analyzer,
         you can specify it here.<br>
        """)
        cleanup = gr.Radio(['Yes', 'No'], value=conf['cleanup'],
                           label='Clean up the created .radio_analyzer folder for the file?')
        reduce_noise = gr.Radio(['Yes', 'No'], value=conf['reduce_noise'],
                                label='Noise reduce: If set to \'Yes\','
                                      ' noisereduce will attempt to reduce noise on the audio file.')
        to_txt = gr.Radio(['Yes', 'No'], value=conf['to_txt'],
                          label='Create .txt: If set to \'Yes\', the transcript and translation will be'
                                ' additionally saved as a .txt file in the folder of the audio file.')
        w_model = gr.Radio(['large-v2', 'large', 'medium', 'small', 'base', 'tiny'],
                           label='Select the Whisper model', value=conf['w_model'])
        t_model = gr.Radio(['Whisper', 'Helsinki', 'Facebook'], label='Select the translationr model',
                           value=conf['t_model'])
        custom = gr.Textbox(
            label='Alter the name for the save file here. If none is given, a default name will be chosen.')
        path = gr.Textbox(label='Adjust your base directory here. Default is: ~/.radio_analyzer',
                          value=conf['base_directory'])

        # Button to save config

        config_button = gr.Button('Safe config', size='lg')
        config_button.click(save_conf, inputs=[cleanup, reduce_noise, to_txt, w_model, t_model, path], outputs=[])

    # Tab for Description

    with gr.Tab('Description'):
        gr.Markdown(
            """
                # App Description

                This app is designed to transcribe, translate, and analyze audio files containing intercepted
                 radio communications from the Russian Armed Forces during the Ukraine War.
                
                ## Input Description:
                
                You have the option to select an audio file using the File Selector or simply by dragging and dropping
                the file into the Input field. If you wish to reload a previous analysis result, please load the
                corresponding JSON file via the 'JSON Selector' tab. Should there be a new result, for instance,
                from utilizing the 'Bulk Analysis' feature, it is necessary to refresh the dropdown using the
                'Refresh Dropdown' button to view the most recent results.
                

                ## Output Fields Description:

                - **Name of the File**: Name of the analyzed audio file.
                - **Path to Audio File**: Location of the audio file.
                - **Whisper Model Used**: The Whisper model used for transcription and translation.
                - **Start Time of Analysis**: Start time of the analysis.
                - **Overall Sentiment**: Sentiment derived from the translated text.
                - **Mood of the Text**: Mood classification of the text. Categories include: 'Aggressive',
                    'Defensive', 'Concerned', and 'Optimistic'.
                - **Tactical Labels**: Text classification for potential strategic content. Categories include: 
                    'Unclassified', 'Logistic and Supplies', 'Casualty Report', 'Reconnaissance activities', 
                    'Troop movement', 'Military strategy discussion', and 'Plans for future operations'.
                - **Legal Labels**: Text classification for potential legal implications. Categories include:
                    'Unclassified', 'Looting', 'Crimes', 'Rape', 'Violation of international law', and 'Pillage'.
                - **NER**: Displays the translated text and highlights all detected names, locations, organizations,
                 and miscellaneous entities.

                Under the 'Original Text' tab, you can find the original transcribed text of the audio file.
            """)

    # Button functionality

    file_selector_button.click(run_and_display, inputs=[obj, w_model, custom, path, reduce_noise, to_txt, cleanup, t_model],
                               outputs=[highlight, sentiment, mood, label_tac, label_legal, org, file_name,
                               file_path, model, time, js, loc_map])

    json_button.click(run_and_display, inputs=[json_obj, w_model, custom, path, reduce_noise, to_txt, cleanup, t_model],
                      outputs=[highlight, sentiment, mood, label_tac, label_legal, org, file_name,
                               file_path, model, time, js, loc_map])

    bulk_button.click(bulk_analysis, inputs=[folder_list, w_model, custom, path, reduce_noise, to_txt, cleanup,
                                             t_model], outputs=[progress])


def run_radio_analyzer():
    webbrowser.open(url='http://127.0.0.1:7860', new=2, autoraise=True)
    analyzer_webapp.queue()
    analyzer_webapp.launch()



