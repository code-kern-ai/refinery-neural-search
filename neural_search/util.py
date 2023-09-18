import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from scipy.spatial.distance import cdist
from typing import Any, Tuple, Union, List, Optional, Dict
from fastapi import status

from submodules.model.business_objects import embedding
from submodules.model.enums import EmbeddingPlatform

from .similarity_threshold import SimilarityThreshold

port = int(os.environ["QDRANT_PORT"])
qdrant_client = QdrantClient(host="qdrant", port=port)

sim_thr = SimilarityThreshold(qdrant_client)

missing_collections_creation_in_progress = False


def most_similar(
    project_id: str,
    embedding_id: str,
    record_id: str,
    limit: int = 100,
    att_filter: Optional[List[Dict[str, Any]]] = None,
    record_sub_key: Optional[int] = None,
):
    embedding_item = embedding.get_tensor(embedding_id, record_id, record_sub_key)
    embedding_tensor = embedding_item.data
    return most_similar_by_embedding(
        project_id, embedding_id, embedding_tensor, limit, att_filter
    )


def most_similar_by_embedding(
    project_id: str,
    embedding_id: str,
    embedding_tensor: List[float],
    limit: int,
    att_filter: Optional[List[Dict[str, Any]]] = None,
    threshold: Optional[float] = None,
) -> List[str]:
    if not is_filter_valid_for_embedding(project_id, embedding_id, att_filter):
        return []
    tmp_limit = limit
    has_sub_key = embedding.has_sub_key(project_id, embedding_id)
    if has_sub_key:
        # new tmp limit to ensure that we get enough results for embedding lists
        # since the limit factor is just an average of embedding list entries rounded up this could be to little depending on the
        # explicit question and the amount of matched sub_keys so we add another 25 to be sure
        limit_factor = embedding.get_qdrant_limit_factor(project_id, embedding_id)
        tmp_limit = (limit * limit_factor) + 25
    query_vector = np.array(embedding_tensor)
    similarity_threshold = threshold
    if similarity_threshold is None:
        similarity_threshold = sim_thr.get_threshold(project_id, embedding_id)
    elif similarity_threshold == -9999:
        similarity_threshold = None
    try:
        search_result = qdrant_client.search(
            collection_name=embedding_id,
            query_vector=query_vector,
            query_filter=__build_filter(att_filter),
            limit=tmp_limit,
            score_threshold=similarity_threshold,
        )
    except Exception:
        return []

    ids = [result.id for result in search_result]
    return embedding.get_match_record_ids_to_qdrant_ids(
        project_id, embedding_id, ids, limit
    )


def is_filter_valid_for_embedding(
    project_id: str,
    embedding_id: str,
    att_filter: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    if not att_filter:
        return True

    embedding_item = embedding.get(project_id, embedding_id)
    filter_attributes = embedding_item.filter_attributes
    if not filter_attributes:
        return False
    for filter_attribute in att_filter:
        if filter_attribute["key"] not in filter_attributes:
            return False

    return True


def __build_filter(att_filter: List[Dict[str, Any]]) -> models.Filter:
    if att_filter is None or len(att_filter) == 0:
        return None
    must = [__build_filter_item(filter_item) for filter_item in att_filter]
    return models.Filter(must=must)


def __build_filter_item(filter_item: Dict[str, Any]) -> models.FieldCondition:
    if isinstance(filter_item["value"], list):
        if filter_item.get("type") == "between":
            return models.FieldCondition(
                key=filter_item["key"],
                range=models.Range(
                    gte=filter_item["value"][0],
                    lte=filter_item["value"][1],
                ),
            )
        else:
            should = [
                models.FieldCondition(
                    key=filter_item["key"], match=models.MatchValue(value=value)
                )
                for value in filter_item["value"]
            ]
            return models.Filter(should=should)
    else:
        return models.FieldCondition(
            key=filter_item["key"],
            match=models.MatchValue(
                value=filter_item["value"],
            ),
        )


def recreate_collection(project_id: str, embedding_id: str) -> int:
    filter_attribute_dict = embedding.get_filter_attribute_type_dict(
        project_id, embedding_id
    )
    all_object = embedding.get_tensors_and_attributes_for_qdrant(
        project_id, embedding_id, filter_attribute_dict
    )
    # note embedding lists use tensor id, others use record ids
    ids, embeddings, payloads = zip(*all_object)
    if len(embeddings) == 0:
        return status.HTTP_404_NOT_FOUND
    vector_size = 0
    if len(embeddings) > 0 and embeddings[0] is not None:
        vector_size = len(embeddings[0])

    qdrant_client.recreate_collection(
        collection_name=embedding_id,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.EUCLID
        ),
    )
    records = None

    if (
        embedding.get(project_id, embedding_id).platform
        == EmbeddingPlatform.PYTHON.value
    ):
        embeddings = [[float(e) for e in embedding] for embedding in embeddings]

    if len(payloads) > 0 and payloads[0] is None:
        records = [
            models.Record(
                id=id,
                vector=embedding,
            )
            for id, embedding in zip(ids, embeddings)
        ]
    else:
        records = [
            models.Record(id=id, vector=e, payload=payload)
            for id, e, payload in zip(ids, embeddings, payloads)
        ]

    qdrant_client.upload_records(collection_name=embedding_id, records=records)
    sim_thr.calculate_threshold(project_id, embedding_id)

    return status.HTTP_200_OK


def get_collections():
    collections = []

    try:
        response = qdrant_client.get_collections()
        collections = [collection.name for collection in response]
    except Exception:
        return collections


def create_missing_collections() -> Tuple[int, Union[List[str], str]]:
    global missing_collections_creation_in_progress

    if missing_collections_creation_in_progress:
        return (
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Another process is already creating missing collections.",
        )

    missing_collections_creation_in_progress = True
    collections = get_collections()
    embedding_items = embedding.get_finished_attribute_embeddings()

    if not embedding_items:
        missing_collections_creation_in_progress = False
        return status.HTTP_412_PRECONDITION_FAILED, "No embeddings found."

    created_collections = []
    for project_id, embedding_id in embedding_items:
        if embedding_id in collections:
            continue

        try:
            recreate_collection(project_id, embedding_id)
            created_collections.append(embedding_id)
        except Exception as e:
            qdrant_client.delete_collection(collection_name=embedding_id)
            print(f"this did not work :(  -> {embedding_id}")
            print(f"Aaaand the error goes to {e}")

    missing_collections_creation_in_progress = False

    return status.HTTP_200_OK, created_collections


def delete_collection(embedding_id: str):
    qdrant_client.delete_collection(collection_name=embedding_id)


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
            status.HTTP_412_PRECONDITION_FAILED,
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

    return status.HTTP_200_OK, [outlier_ids.tolist(), outlier_scores.tolist()]
