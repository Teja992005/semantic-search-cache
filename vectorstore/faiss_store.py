import faiss
import numpy as np


class FAISSStore:

    def __init__(self, embeddings, documents):

        print("Building FAISS index...")

        self.documents = documents

        dimension = embeddings.shape[1]

        # create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # convert embeddings to float32
        embeddings = np.array(embeddings).astype("float32")

        # add embeddings to index
        self.index.add(embeddings)

        print("Total documents indexed:", self.index.ntotal)

    def search(self, query_embedding, k=5):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for i in indices[0]:
            results.append(self.documents[i])

        return results