import numpy as np
import os
from qdrant_client import QdrantClient
from scipy.spatial.distance import cdist
from typing import Any, Tuple, Union, List

from submodules.model.business_objects import embedding

from .similarity_threshold import SimilarityThreshold

port = int(os.environ["QDRANT_PORT"])
qdrant_client = QdrantClient(host="qdrant", port=port)

sim_thr = SimilarityThreshold(qdrant_client)

missing_collections_creation_in_progress = False


def most_similar(project_id: str, embedding_id: str, record_id: str, limit: int = 100):
    embedding_item = embedding.get_tensor(embedding_id, record_id)
    embedding_tensor = embedding_item.data
    return most_similar_by_embedding(project_id, embedding_id, embedding_tensor, limit)


def most_similar_by_embedding(
    project_id: str, embedding_id: str, embedding_tensor: List[float], limit: int
) -> List[str]:
    query_vector = np.array(embedding_tensor)
    similarity_threshold = sim_thr.get_threshold(project_id, embedding_id)
    search_result = qdrant_client.search(
        collection_name=embedding_id,
        query_vector=query_vector,
        query_filter=None,
        limit=limit,
        score_threshold=similarity_threshold,
    )
    similar_records = [result.id for result in search_result]
    return similar_records


def recreate_collection(project_id: str, embedding_id: str) -> int:
    tensors = embedding.get_tensors_by_embedding_id(embedding_id)
    ids, embeddings = zip(*tensors)
    embeddings = np.array(embeddings)

    if len(embeddings) == 0:
        return 404

    vector_size = embeddings.shape[-1]

    qdrant_client.recreate_collection(
        collection_name=embedding_id, vector_size=vector_size, distance="Euclid"
    )
    qdrant_client.upload_collection(
        collection_name=embedding_id, vectors=embeddings, payload=None, ids=ids
    )

    sim_thr.calculate_threshold(project_id, embedding_id)

    return 200


def get_collections():
    response = qdrant_client.openapi_client.collections_api.get_collections()
    collections = [collection.name for collection in response.result.collections]
    return collections


def create_missing_collections() -> Tuple[int, Union[List[str], str]]:
    global missing_collections_creation_in_progress

    if missing_collections_creation_in_progress:
        return 429, "Another process is already creating missing collections."

    missing_collections_creation_in_progress = True

    response = qdrant_client.openapi_client.collections_api.get_collections()
    collections = [collection.name for collection in response.result.collections]
    embedding_items = embedding.get_finished_attribute_embeddings()

    if not embedding_items:
        missing_collections_creation_in_progress = False
        return 412, "No embeddings found."

    created_collections = []
    for project_id, embedding_id in embedding_items:

        if embedding_id in collections:
            continue

        try:
            recreate_collection(project_id, embedding_id)
            created_collections.append(embedding_id)
        except Exception as e:
            qdrant_client.delete_collection(embedding_id)
            print(f"this did not work :(  -> {embedding_id}")
            print(f"Aaaand the error goes to {e}")

    missing_collections_creation_in_progress = False

    return 200, created_collections


def delete_collection(embedding_id: str):
    qdrant_client.delete_collection(embedding_id)


def detect_outliers(
    project_id: str, embedding_id: str, limit: int = 100
) -> Tuple[int, Union[List[Any], str]]:
    labeled_tensors = embedding.get_manually_labeled_tensors_by_embedding_id(
        project_id, embedding_id
    )
    labeled_ids, labeled_embeddings = zip(*labeled_tensors)
    labeled_embeddings = np.array(labeled_embeddings)

    if len(labeled_ids) < 1:
        return (
            412,
            "At least one record must be labeled manually to create outlier slice.",
        )

    unlabeled_tensors = embedding.get_not_manually_labeled_tensors_by_embedding_id(
        project_id, embedding_id
    )
    unlabeled_ids, unlabeled_embeddings = zip(*unlabeled_tensors)
    unlabeled_embeddings = np.array(unlabeled_embeddings)

    outlier_scores = np.sum(
        cdist(labeled_embeddings, unlabeled_embeddings, "euclidean"), axis=0
    )
    sorted_index = np.argsort(
        outlier_scores,
        axis=None,
    )[::-1]

    max_records = min(round(0.05 * len(sorted_index)), limit)

    outlier_ids = np.array(unlabeled_ids)[sorted_index[:max_records]]
    outlier_scores = outlier_scores[sorted_index[:max_records]]

    return 200, [outlier_ids.tolist(), outlier_scores.tolist()]
