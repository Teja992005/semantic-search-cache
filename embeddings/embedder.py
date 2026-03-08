from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, documents):

        print("Creating embeddings for documents...")

        embeddings = self.model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, query):

        embedding = self.model.encode([query])[0]

        return embedding