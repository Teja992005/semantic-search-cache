from fastapi import FastAPI
from pydantic import BaseModel

from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FAISSStore
from clustering.fuzzy_cluster import FuzzyCluster
from cache.semantic_cache import SemanticCache


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


print("Starting Semantic Search System...")

print("Loading dataset...")
documents = load_dataset()
print(f"Loaded {len(documents)} documents")

print("Loading embedding model...")
embedder = Embedder()

print("Generating document embeddings...")
embeddings = embedder.embed_documents(documents)

print("Building FAISS vector index...")
vector_store = FAISSStore(embeddings, documents)

print("Running fuzzy clustering...")
cluster_model = FuzzyCluster(embeddings, n_clusters=10)

print("Initializing semantic cache...")
cache = SemanticCache(threshold=0.8)

print("System ready. API is live.")


@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query
    print(f"Received query: {query}")
    query_embedding = embedder.embed_query(query)

    cached, similarity = cache.lookup(query_embedding)

    if cached:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["query"],
            "similarity_score": float(similarity),
            "result": cached["result"][:500],
            "dominant_cluster": cached["cluster"]
        }

    results = vector_store.search(query_embedding, k=1)

    result_text = results[0]

    cluster_id = cluster_model.get_dominant_cluster(0)

    cache.add(query, query_embedding, result_text, cluster_id)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_text[:500],
        "dominant_cluster": cluster_id
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}