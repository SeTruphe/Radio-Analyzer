import shutil

import gradio as gr
import radio_analyzer
import json
import os
import ast
import webbrowser


def run_app(obj, w_model, custom, path, reduce_noise, to_txt, clean_up):
    """
    :param obj: _TemporaryFileWrapper-Object which is created by Gradio and hast the path to the Audiofile
    :param custom: Custom name for the folder where audio chunks, transcriptions, and translations are stored.
        If not provided, a default name is generated.
    :param path: Root directory where analysis folders are created. Can be overridden with a custom path.
    :return: Returns data formatted for display in the Gradio app.
    """

    path_to_audio = obj.name

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
        reduce_noise = None
    else:
        reduce_noise = True

    if to_txt == 'No':
        to_txt = False
    else:
        to_txt = True
        clean_up = False

    # Get file format

    path_to_audio = path_to_audio.replace('\"', '')

    _, file_format = os.path.splitext(path_to_audio)
    file_format = file_format[1:]

    # If json, load file. If mp3, analyse file. Else return error

    if file_format == 'json':
        with open(path_to_audio, 'r') as jfile:
            data = json.load(jfile)

    elif file_format == 'mp3' or file_format == '.wav':
        data = radio_analyzer.radio_analyzer(path_to_audio,
                                             whisper_model=w_model,
                                             clean_up=clean_up,
                                             custom_name=custom,
                                             base_path=path,
                                             reduce_noise=reduce_noise,
                                             to_txt=to_txt
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
    shutil.rmtree(os.path.dirname(path_to_audio))
    shutil.rmtree(os.path.dirname(obj.orig_name))

    return {'text': english,
            'entities': ner}, data_sentiment, mood_data, labels_tactical, labels_legal, original, name, audio_path, whisper_model, ctime, data


# Create Gradio App

with gr.Blocks() as analyzer_webapp:
    gr.Image(value='..\\data\\media\\wordmark-bright.svg', height=100, container=False)

    # Creates tab for path input

    with gr.Tab('Analyze'):
        obj = gr.File(label='Please input the Audio file you want to Analyze or an JSON'
                            ' file from a previously analyzed Audio file')

        # Creates graphic output for the results

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

        with gr.Tab('Original Text'):
            org = gr.components.Textbox(label='Original Text')

        with gr.Tab('Raw Data'):
            js = gr.JSON(label="Raw JSON data")

        with gr.Tab('Description'):
            gr.Markdown(
                """
                    # App Description
    
                    This app is designed to transcribe, translate, and analyze audio files containing intercepted radio communications from the Russian Armed Forces during the Ukraine War.
    
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

    # Creates tab for advanced parameters

    with gr.Tab('Advanced Settings'):
        gr.Markdown("""
        In this tab, you can adjust and input the advanced settings of the app.
        Clean Up: Enable this option to automatically delete the newly created folder in the .radio_analyzer directory for the current audio file after the analysis process is complete.<br><br>
        Create .txt Files: Enable this option to save the transcriptions and translations as text files within the .radio_analyzer folder designated for the audio file.<br>
        Whisper Model: Use this setting to select the Whisper model you'd like to use for analysis. The default model is large-v2.<br>
        Save File Name: Specify a custom name for your save file. This name will also serve as the folder name within the .radio_analyzer directory.<br>
        Adjust Base Directory: If you wish to use a different base directory for .radio_analyzer, you can specify it here.<br>
        """)
        cleanup = gr.Radio(['Yes', 'No'], value='No', label='Clean up the created .radio_analyzer folder for the file?')
        reduce_noise = gr.Radio(['Yes', 'No'], value='No',
                                label='Noise reduce: If set to \'Yes\', noisereduce will attempt to reduce noise on the audio file.')
        to_txt = gr.Radio(['Yes', 'No'], value='No',
                          label='Create .txt: If set to \'Yes\', the transcript and translation will be additionally saved as a .txt file in the folder of the audio file.')
        w_model = gr.Radio(['large-v2', 'large', 'medium', 'small', 'base', 'tiny'], label='Select the Whisper model',
                           value='large-v2')
        custom = gr.Textbox(
            label='Alter the name for the save file here. If none is given, a default name will be chosen.')
        path = gr.Textbox(label='Adjust your base directory here. Default is: ~/.radio_analyzer', placeholder='')

    # Creates Button which triggers the analysis process

    text_button = gr.Button('Analyze', size='lg')
    text_button.click(run_app, inputs=[obj, w_model, custom, path, reduce_noise, to_txt, cleanup],
                      outputs=[highlight, sentiment, mood, label_tac, label_legal, org,
                               file_name, file_path, model, time, js])

if __name__ == '__main__':
    webbrowser.open(url='http://127.0.0.1:7860', new=2, autoraise=True)
    analyzer_webapp.launch()

