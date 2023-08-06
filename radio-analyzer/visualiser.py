import gradio as gr
import radio_analyzer
import json
import os
import ast


def run_app(path_to_audio, custom_name=None, clean=False, base_path=None):

    """
    :param path_to_audio: path to the audiofile you want to analyse
    :param custom_name: The app creates a folder for the audio chunks as well as transcription and translation text
            files. The folder name is generated automatically. You can alter the folder name here.
    :param clean: If set to true, generated folder for the project is deleted after the analysis.
            Set to true to safe space. Default is false.
    :param base_path: The folders for the analysis are generated in the base path of the user.
            You can define a different path here.
    :return: returns the data in a formate suitable for the Gradio app
    """

    # Transfer Clean up string into boolean

    if clean == 'Yes':
        clean_up = True
    else:
        clean_up = False

    # Clean empty inputs

    if base_path == "":
        base_path = os.path.join('~', '.radio_analyzer')

    if custom_name == "":
        custom_name = None

    # Get file format

    path_to_audio = path_to_audio.replace('\"', '')

    _, file_format = os.path.splitext(path_to_audio)
    file_format = file_format[1:]

    # If json, load file. If mp3, analyse file. Else return error

    if file_format == "json":
        with open(path_to_audio, 'r') as jfile:
            data = json.load(jfile)

    elif file_format == "mp3":
        data = radio_analyzer.radio_analyzer(path_to_audio,
                                             clean_up=clean_up,
                                             custom_name=custom_name,
                                             base_path=base_path)

    else:
        print('Wrong file format')
        return None

    data_sentiment = data['sentiment']
    ner = ast.literal_eval(data['ner'])

    for entity in ner:
        entity['entity'] = entity.pop('entity_group')

    labels_tactical = {lab: conf for lab, conf in data['labels_tactical']}
    labels_legal = {lab: conf for lab, conf in data['labels_legal']}
    major_tac = data['major_tactical']['label']
    major_legal = data['major_legal']['label']
    mood_data = data['label_mood']['label']
    english = data['english']
    original = data['original']
    name = data['file_name']
    audio_path = data['path']
    whisper_model = data['model']
    ctime = data['time_of_analysis']

    return {'text': english, 'entities': ner}, data_sentiment, mood_data, major_tac, labels_tactical, major_legal, labels_legal, original, name, audio_path, whisper_model, ctime


# Create Gradio App

with gr.Blocks() as analyzer_webapp:
    gr.Image(value='..\\data\\media\\wordmark-bright.svg', height=100, container=False)

    # Creates tab for path input

    with gr.Tab('Analyze'):
        path_input = gr.Textbox(label='Please input the local path to the Audio file you want to Analyze')

        # Creates graphic output for the results

        with gr.Tab('Analysis Data'):
            with gr.Row():
                file_name = gr.components.Textbox(label='Name of the file')
                file_path = gr.components.Textbox(label='Path to Audio file')
            with gr.Row():
                model = gr.components.Textbox(label='Used Whisper-Model')
                time = gr.components.Textbox(label='Start time of Analysis')
            with gr.Row():
                sentiment = gr.components.Textbox(label='Overall Sentiment')
                mood = gr.components.Textbox(label='Mood of the Text')
            with gr.Row():
                maj_tac = gr.components.Textbox(label='Majority tactical label')
                maj_legal = gr.components.Textbox(label='Majority legal label')
            with gr.Row():
                label_tac = gr.Label(label='All tactical labels above 20%')
                label_legal = gr.Label(label='All legal labels above 20%')

            highlight = gr.HighlightedText(label='NER')

        with gr.Tab('Original Text'):
            org = gr.components.Textbox(label='Original Text')

    # Creates tab for advanced parameters

    with gr.Tab('Advanced Settings'):
        gr.Markdown("""
        In this tab you can adjust and input the advanced settings of the app.
        Please visit https://github.com/SeTruphe/Radio-Analyzer for further information's on the advanced settings
        """)
        cleanup = gr.Radio(['Yes'], label='Cleanup the chunks after the process')
        custom = gr.Textbox(label='Alter the name for the Savefile here. If none is given, a default name is chosen')
        path = gr.Textbox(label='Adjust your base directory here. Default is: ~/.radio_analyzer', placeholder='')

    # Creates Button which triggers the analysis process

    text_button = gr.Button('Analyze', size='lg')
    text_button.click(run_app, inputs=[path_input, custom, cleanup, path],
                      outputs=[highlight, sentiment, mood, maj_tac, label_tac,maj_legal, label_legal, org, file_name, file_path, model, time])


if __name__ == '__main__':
    analyzer_webapp.launch()
