import gradio as gr
import os
from src.gen_dict import generate_translation_dictionary
from src.translate import translate_chinese_to_amis
from src.get_model import faiss_index_path, amis_mappings_path


def main(word):

    # Check if the FAISS index and Amis mappings exist
    if not os.path.exists(faiss_index_path) or not os.path.exists(
        os.environ.get(amis_mappings_path)
    ):
        generate_translation_dictionary()

    # Example usage
    amis_translation = translate_chinese_to_amis(word)

    return amis_translation


client = gr.Interface(fn=main, inputs="text", outputs="text", title="Amis Translator")

client.launch()
