from sentence_transformers import SentenceTransformer


E5_MODEL = "intfloat/multilingual-e5-base"


def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_documents(
    model: SentenceTransformer,
    model_name: str,
    texts: list[str],
):
    if model_name == E5_MODEL:
        texts = [f"passage: {text}" for text in texts]

    return model.encode_document(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_queries(
    model: SentenceTransformer,
    model_name: str,
    texts: list[str],
):
    if model_name == E5_MODEL:
        texts = [f"query: {text}" for text in texts]

    return model.encode_query(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )