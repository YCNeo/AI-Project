import gradio as gr
import os
from src.gen_dict import generate_translation_dictionary
from src.translate import translate_chinese_to_amis


def main(word: str):
    # Check if the FAISS index and Amis mappings exist
    if not os.path.exists(os.environ.get("FAISS_INDEX_PATH")) or not os.path.exists(
        os.environ.get("AMIS_MAPPINGS_PATH")
    ):
        print("Translation dictionary not found. Generating now...")
        generate_translation_dictionary()
    else:
        print("Translation dictionary found. Proceeding to translation.")

    # Example usage
    amis_translation = translate_chinese_to_amis(word)
    print(f'The Amis translation of "{word}" is "{amis_translation}".')


demo = gr.Interface(fn=main, inputs="text", outputs="text")

demo.launch()
