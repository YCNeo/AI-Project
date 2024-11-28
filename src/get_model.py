import os
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


# faiss_index_path = os.environ.get("FAISS_INDEX_PATH")
faiss_index_path = "/app/doc/zh_embeddings.index"
# amis_mappings_path = os.environ.get("AMIS_MAPPINGS_PATH")
amis_mappings_path = "/app/doc/amis_mappings.pkl"
