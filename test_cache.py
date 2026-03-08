from embeddings.embedder import Embedder
from cache.semantic_cache import SemanticCache

embedder = Embedder()

cache = SemanticCache()

query1 = "space rocket launch"
query2 = "how rockets launch into space"

emb1 = embedder.embed_query(query1)
emb2 = embedder.embed_query(query2)

cache.add(query1, emb1, "Result about rockets", 3)

result, similarity = cache.lookup(emb2)

print("Similarity:", similarity)

if result:
    print("CACHE HIT")
else:
    print("CACHE MISS")