import gradio as gr
import radio_analyzer
import json
import os
import ast


def run_app(path_to_audio, w_model, custom, cleanup, path, reduce_noise):

    """
    :param path_to_audio: Path to the target audio file for analysis.
    :param custom: Custom name for the folder where audio chunks, transcriptions, and translations are stored.
        If not provided, a default name is generated.
    :param cleanup: If True, the generated project folder is deleted post-analysis to conserve space. Default is False.
    :param path: Root directory where analysis folders are created. Can be overridden with a custom path.
    :return: Returns data formatted for display in the Gradio app.
    """

    # Transfer Clean up string into boolean

    if cleanup == 'No':
        clean_up = False
    else:
        clean_up = True

    # Clean empty inputs

    if path == '':
        path = os.path.join('~', '.radio_analyzer')

    if custom == '':
        custom = None

    if w_model == '':
        w_model = 'large-v2'

    if reduce_noise == '':
        reduce_noise = None

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
                                             whisper_model=w_model,
                                             clean_up=clean_up,
                                             custom_name=custom,
                                             base_path=path,
                                             reduce_noise=reduce_noise
                                             )

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

    return {'text': english, 'entities': ner}, data_sentiment, mood_data, labels_tactical, labels_legal, original, name, audio_path, whisper_model, ctime, data


# Create Gradio App

with gr.Blocks() as analyzer_webapp:
    gr.Image(value='..\\data\\media\\wordmark-bright.svg', height=100, container=False)

    # Creates tab for path input

    with gr.Tab('Analyze'):
        path_input = gr.Textbox(label='Please input the local path to the Audio file you want to Analyze')

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

    # Creates tab for advanced parameters

    with gr.Tab('Advanced Settings'):
        gr.Markdown("""
        In this tab, you can adjust and input the advanced settings of the app.
        Please visit https://github.com/SeTruphe/Radio-Analyzer for further information on the advanced settings.
        """)
        cleanup = gr.Radio(['No'], label='Clean up the chunks after the process?')
        reduce_noise = gr.Radio(['Yes'],
                                label='Noise reduce: If set to \'Yes\', noisereduce will attempt to reduce noise on the audio file.')
        to_txt = gr.Radio(['Yes'],
                          label='Create .txt: If set to \'Yes\', the transcript and translation will be additionally saved as a .txt file in the folder of the audio file.')
        w_model = gr.Radio(['large-v2', 'large', 'medium', 'small', 'base', 'tiny'], label='Select the Whisper model',
                           value='large-v2')
        custom = gr.Textbox(
            label='Alter the name for the save file here. If none is given, a default name will be chosen.')
        path = gr.Textbox(label='Adjust your base directory here. Default is: ~/.radio_analyzer', placeholder='')


    # Creates Button which triggers the analysis process

    text_button = gr.Button('Analyze', size='lg')
    text_button.click(run_app, inputs=[path_input, w_model, custom, cleanup, path, reduce_noise],
                      outputs=[highlight, sentiment, mood, label_tac, label_legal, org,
                               file_name, file_path, model, time, js])


if __name__ == '__main__':
    analyzer_webapp.launch()
