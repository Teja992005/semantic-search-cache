from data.dataset_loader import load_dataset
from embeddings.embedder import Embedder
from clustering.fuzzy_cluster import FuzzyCluster

docs = load_dataset()

embedder = Embedder()

embeddings = embedder.embed_documents(docs[:500])

cluster = FuzzyCluster(embeddings, n_clusters=10)

print("Dominant cluster for document 0:", cluster.get_dominant_cluster(0))