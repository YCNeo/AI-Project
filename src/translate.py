import os
import numpy as np
import faiss
import pickle
import torch
from concurrent.futures import ThreadPoolExecutor
from .get_model import get_model, get_tokenizer, faiss_index_path, amis_mappings_path


def translate_chinese_to_amis(chinese_word):
    """
    Translates a Chinese word to Amis using the precomputed FAISS index and mappings.
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

    def compute_embedding():
        with torch.no_grad():
            inputs = tokenizer(
                [chinese_word], padding=True, truncation=True, return_tensors="pt"
            )
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            # Normalize embedding
            query_embedding = query_embedding / np.linalg.norm(
                query_embedding, axis=1, keepdims=True
            )
        return query_embedding

    def search_faiss(query_embedding):
        distances, indices = index.search(query_embedding, k=1)
        idx = indices[0][0]
        distance = distances[0][0]
        # Set a threshold for the distance to filter out irrelevant results
        threshold = 0.45  # Adjust this threshold based on your requirements
        if distance > threshold:
            return "Translation not found"
        if idx < len(amis_mappings):
            return amis_mappings[idx]
        else:
            return "Translation not found"

    with ThreadPoolExecutor() as executor:
        future_embedding = executor.submit(compute_embedding)
        query_embedding = future_embedding.result()
        future_search = executor.submit(search_faiss, query_embedding)
        amis_translation = future_search.result()

    return amis_translation
