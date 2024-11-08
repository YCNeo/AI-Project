import gradio as gr
from src.ModelCall import model_call


def translate(word):
    return model_call(word)


demo = gr.Interface(fn=translate, inputs="text", outputs="text")

demo.launch()
