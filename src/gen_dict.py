import os
import numpy as np
import pandas as pd
import faiss
import pickle
import torch
from numpy.linalg import norm
from .get_model import get_model, get_tokenizer, faiss_index_path, amis_mappings_path


def generate_translation_dictionary():
    """
    Generates the translation dictionary by computing embeddings,
    building the FAISS index, and saving the data to disk.
    """
    df = pd.read_excel(os.environ.get("DOC_PATH"))
    zh_words = df["zhi"].tolist()
    amis_words = df["amis"].tolist()

    # Ensure the lists are of the same length
    assert len(zh_words) == len(
        amis_words
    ), "The word lists must be of the same length."

    print("Generating translation dictionary...")

    # Initialize the tokenizer and model
    tokenizer = get_tokenizer()
    model = get_model()
    model.eval()

    # Function to get embeddings
    def get_embeddings(text_list):
        with torch.no_grad():
            inputs = tokenizer(
                text_list, padding=True, truncation=True, return_tensors="pt"
            )
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            embeddings = embeddings.numpy()
            # Normalize embeddings
            embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)
            return embeddings

    # Compute embeddings for Chinese words
    zh_embeddings = get_embeddings(zh_words)

    # Build a FAISS index with the Chinese embeddings
    dimension = zh_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(zh_embeddings)

    # Save the FAISS index to disk
    faiss.write_index(index, faiss_index_path)

    # Store the mapping from index positions to Amis words
    amis_mappings = [amis_words[i] for i in range(len(amis_words))]

    # Save the Amis mappings to disk
    with open(amis_mappings_path, "wb") as f:
        pickle.dump(amis_mappings, f)

    print("Translation dictionary generated and saved to disk.")
