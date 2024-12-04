import os
from transformers import AutoModel, AutoTokenizer


def get_model():
    return AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-zh",
        trust_remote_code=True,
    )


def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v2-base-zh", trust_remote_code=True
    )


faiss_index_path = "/app/doc/zh_embeddings.index"
amis_mappings_path = "/app/doc/amis_mappings.pkl"
