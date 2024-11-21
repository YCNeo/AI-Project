import torch
from transformers import AutoModel, AutoTokenizer


def get_model():
    return AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-zh",
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
    )


def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v2-base-zh", trust_remote_code=True
    )
