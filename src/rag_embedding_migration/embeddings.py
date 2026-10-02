from sentence_transformers import SentenceTransformer

def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True)

def embed_documents(model: SentenceTransformer, texts: list[str]):
    return model.encode_document(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

def embed_queries(model: SentenceTransformer, texts: list[str]):
    return model.encode_query(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )