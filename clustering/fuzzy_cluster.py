import numpy as np
import skfuzzy as fuzz


class FuzzyCluster:

    def __init__(self, embeddings, n_clusters=10):

        print("Performing fuzzy clustering...")

        self.embeddings = np.array(embeddings)

        # transpose for scikit-fuzzy
        data = self.embeddings.T

        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            c=n_clusters,
            m=2,
            error=0.005,
            maxiter=1000
        )

        self.centers = cntr
        self.membership = u

        print("Fuzzy clustering complete")

    def get_dominant_cluster(self, doc_index):

        memberships = self.membership[:, doc_index]

        return int(np.argmax(memberships))