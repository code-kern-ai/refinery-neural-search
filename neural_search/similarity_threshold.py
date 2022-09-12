import random
from typing import List
import numpy as np
from scipy.spatial.distance import cdist

from submodules.model.business_objects import embedding


class SimilarityThreshold:
    """
    Calculates and stores the threshold for the similarity search.
    """

    def __init__(self, qdrant_client) -> None:
        """
        Expects qdrant client.
        In the threshold dictionary for each embedding the threshold value is stored.
        """
        self.qdrant_client = qdrant_client

    def get_threshold(self, project_id: str, embedding_id: str) -> float:
        """
        Returns the threshold for the given embedding if already existing.
        Otherwise the threshold is calculated.
        """
        threshold = embedding.get(project_id, embedding_id).similarity_threshold
        if threshold is None:
            threshold = self.calculate_threshold(project_id, embedding_id)
        return threshold

    def calculate_threshold(
        self,
        project_id: str,
        embedding_id: str,
        percentile: int = 5,
        limit: int = 500,
    ) -> None:
        """
        Calculates the threshold on a sample of the embedding's records.
        The threshold is written to the database.

        Args:
            embedding_id (str)
            percentile (int): percentile which should be used to define the threshold.
            limit (int): maximum numbers of records in sample.
        """
        scores = self.get_scores(embedding_id, limit)
        threshold = np.percentile(scores, percentile)
        embedding.update_similarity_threshold(
            project_id, embedding_id, threshold, with_commit=True
        )
        return threshold

    def get_scores(
        self, embedding_id: str, limit: int = 500, distance: str = "euclidean"
    ) -> List[float]:
        """
        Calculates the pairwise distances for a sub sample of the embedding's records.

        Args:
            embedding_id (str)
            limit (int): maximum numbers of records in sample.
        Returns:
            List[float]: containing the pairwise distances
        """
        record_ids = embedding.get_record_ids_by_embedding_id(embedding_id)

        if len(record_ids) < limit:
            sample_ids = record_ids
        else:
            sample_ids = random.sample(record_ids, limit)

        sample_tensors = embedding.get_tensors_by_record_ids(embedding_id, sample_ids)
        sample_ids, sample_embeddings = zip(*sample_tensors)
        sample_embeddings = np.array(sample_embeddings)

        idx = np.triu_indices(sample_embeddings.shape[0], 1)
        scores = cdist(sample_embeddings, sample_embeddings, metric=distance)[idx]

        return scores
