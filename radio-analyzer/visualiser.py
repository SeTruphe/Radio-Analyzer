import gradio as gr
import radio_analyzer
import json
import os


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

    if clean == "Yes":
        clean_up = True
    else:
        clean_up = False

    # Get file format

    path_to_audio = path_to_audio.replace("\"", "")

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
        print("Wrong file format")
        return None

    data_sentiment = data["sentiment"]
    ner = data["ner"]
    labels = data["labels"]
    english = data["english"]
    original = data["original"]

    return {"text": english, "entities": ner}, data_sentiment, labels, original


# Create Gradio App

with gr.Blocks() as analyzer_webapp:
    gr.Markdown("Input your Path and Parameters here")

    # Creates tab for path input

    with gr.Tab("Path"):
        path_input = gr.Textbox(label="Absolute Path to Audiofile")

    # Creates tab for advanced parameters

    with gr.Tab("Advanced Settings"):
        gr.Markdown("Cleanup: Audio chunks and Transkript/Translation files will be deleted after operation")
        cleanup = gr.Radio(["Yes"], label="Cleanup the chunks after the process")
        custom = gr.Textbox(label="Alter the name for the Savefile here. If none is given, a default name is chosen")
        path = gr.Textbox(label="Adjust your base directory here. Default is: ~/.radio_analyzer")

    # Creates graphic output for the results

    highlight = gr.Highlightedtext(label="NER")
    label = gr.Label(label="Label")
    sentiment = gr.components.Textbox()
    org = gr.components.Textbox()

    # Creates Button which triggers the analysis process

    text_button = gr.Button("Analyze")
    text_button.click(run_app, inputs=[path_input, custom, cleanup, path], outputs=[highlight, sentiment, label, org])


if __name__ == '__main__':
    analyzer_webapp.launch()
