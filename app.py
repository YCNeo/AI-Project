import gradio as gr
import os
import jieba
from src.gen_dict import generate_translation_dictionary
from src.translate import translate_chinese_to_amis
from src.get_model import faiss_index_path, amis_mappings_path

# Check if the FAISS index and Amis mappings exist
if not os.path.exists(faiss_index_path) or not os.path.exists(
    os.environ.get(amis_mappings_path)
):
    generate_translation_dictionary()


def main(sentence):
    print("Received sentence:", sentence)
    results = f"Received sentence: {sentence}\n\nTranslation:\n"
    return translate_chinese_to_amis(sentence, results)


client = gr.Interface(fn=main, inputs="text", outputs="text", title="Amis Translator")

client.launch()
