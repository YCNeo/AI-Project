import os
import numpy as np
import faiss
import pickle
import torch
import jieba
from pypinyin import pinyin, Style
from concurrent.futures import ThreadPoolExecutor
from .get_model import get_model, get_tokenizer, faiss_index_path, amis_mappings_path


def translate_chinese_to_amis(chinese_sentence, results):
    """
    Translates a Chinese sentence to Amis using the precomputed FAISS index and mappings.
    """
    # Load the FAISS index
    index = faiss.read_index(faiss_index_path)

    # Load the Amis mappings
    with open(amis_mappings_path, "rb") as f:
        amis_mappings = pickle.load(f)

    # Initialize the tokenizer and model
    tokenizer = get_tokenizer()
    model = get_model()
    model.eval()

    def compute_embedding(text):
        with torch.no_grad():
            inputs = tokenizer(
                [text], padding=True, truncation=True, return_tensors="pt"
            )
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding

    def search_faiss(query_embedding):
        distances, indices = index.search(query_embedding, k=1)
        idx = indices[0][0]
        distance = distances[0][0]
        return idx, distance

    # Compute embedding for the entire sentence
    sentence_embedding = compute_embedding(chinese_sentence)
    sentence_idx, sentence_distance = search_faiss(sentence_embedding)

    # Set a threshold for the distance to filter out irrelevant results
    threshold = 0.8

    if sentence_distance < threshold:
        # Split the sentence into words and process each word
        words = jieba.lcut(chinese_sentence)
        for word in words:
            if word.strip():
                word_embedding = compute_embedding(word)
                word_idx, word_distance = search_faiss(word_embedding)
                if word_distance < threshold:
                    translation = get_pinyin(word)
                else:
                    translation = (
                        amis_mappings[word_idx]
                        if word_idx < len(amis_mappings)
                        else get_pinyin(word)
                    )
                results += f"{word}: {translation}\n"
    else:
        if sentence_idx < len(amis_mappings):
            results += f"\t{chinese_sentence}:\t{amis_mappings[sentence_idx]}\n"
        else:
            results += f"\t{chinese_sentence}:\t{get_pinyin(chinese_sentence)}\n"

    return results


def get_pinyin(word):
    roman_pin_yin = "[roman_pinyin] "
    pinyin_list = pinyin(word, style=Style.TONE3)
    return roman_pin_yin + " ".join([item[0] for item in pinyin_list])
