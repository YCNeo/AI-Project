import os
import numpy as np
import faiss
import pickle
import torch
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

    # Compute embedding for the input Chinese word
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

    # Search for the nearest embedding
    distances, indices = index.search(query_embedding, k=1)
    idx = indices[0][0]

    # Retrieve the corresponding Amis word
    if idx < len(amis_mappings):
        amis_translation = amis_mappings[idx]
    else:
        amis_translation = "Translation not found"
    return amis_translation
