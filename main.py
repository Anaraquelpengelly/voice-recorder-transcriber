import gradio as gr
from utils.transcription_functions import response

## set the gradio theme
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

with gr.Blocks() as demo:
    title = gr.Markdown("# Voice Recorder and Transcriber")
    description = gr.Markdown("Record audio or upload a voice recording for transcription.")
    filename_input = gr.Textbox(label="Desired Output Filename", max_lines=1)
    input_mode = gr.Radio(["Microphone", "Upload"], label="Select Input Type", value="Microphone")
    mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Speak now")
    upload_input = gr.Audio(sources="upload", type="numpy", label="Or upload a voice recording")
    output_txt = gr.Textbox(label="Transcription")

    # Toggle visibility depending on input_mode selection
    def toggle_inputs(mode):
        return gr.update(visible=mode=="Microphone"), gr.update(visible=mode=="Upload")
    input_mode.change(toggle_inputs, input_mode, outputs=[mic_input, upload_input])

    # Bind the response function to each input, so it runs when audio changes
    mic_input.change(response, [mic_input, filename_input], [output_txt])
    upload_input.change(response, [upload_input, filename_input], [output_txt])


if __name__ == "__main__":
    demo.launch(debug=True)