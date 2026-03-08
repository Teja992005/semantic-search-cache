from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder

docs = load_dataset()

embedder = Embedder()

embeddings = embedder.embed_documents(docs[:100])

print("Embedding shape:", embeddings.shape)