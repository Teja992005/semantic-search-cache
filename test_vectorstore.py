from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore

# load documents
docs = load_dataset()

# create embedder
embedder = Embedder()

# create embeddings for first 500 docs (for faster test)
embeddings = embedder.embed_documents(docs[:500])

# create vector store
store = FAISSStore(embeddings, docs[:500])

# test query
query = "space rocket launch"

query_embedding = embedder.embed_query(query)

results = store.search(query_embedding)

print("\nTop search results:\n")

for r in results:
    print(r[:200])
    print("------")