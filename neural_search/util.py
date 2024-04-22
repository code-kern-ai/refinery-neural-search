import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from scipy.spatial.distance import cdist
from typing import Any, Tuple, Union, List, Optional, Dict
from fastapi import status

from submodules.model.business_objects import (
    embedding,
    record_label_association,
    record,
)
from submodules.model.enums import EmbeddingPlatform, LabelSource

from .similarity_threshold import SimilarityThreshold

port = int(os.environ["QDRANT_PORT"])
qdrant_client = QdrantClient(host="qdrant", port=port, timeout=60)

sim_thr = SimilarityThreshold(qdrant_client)

missing_collections_creation_in_progress = False

LABELS_QDRANT = "@@labels@@"


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
        # no filter attributes => with filter requested results in an empty list since no record can match the value
        return False
    for filter_attribute in att_filter:
        if filter_attribute["key"] not in filter_attributes and not __is_label_filter(
            filter_attribute["key"]
        ):
            return False

    return True


def __is_label_filter(key: str) -> bool:
    parts = key.split(".")
    if len(parts) == 1:
        return False
    return parts[0] == LABELS_QDRANT


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
    embedding_item = embedding.get(project_id, embedding_id)
    if not embedding_item:
        return status.HTTP_404_NOT_FOUND
    filter_attribute_dict = embedding.get_filter_attribute_type_dict(
        project_id, embedding_id
    )
    all_object = embedding.get_tensors_and_attributes_for_qdrant(
        project_id, embedding_id, filter_attribute_dict
    )
    # note embedding lists use tensor id, others use record ids
    record_ids, embeddings, payloads, tensor_ids = zip(*all_object)
    if len(embeddings) == 0:
        return status.HTTP_404_NOT_FOUND
    vector_size = 0
    if len(embeddings) > 0 and embeddings[0] is not None:
        vector_size = len(embeddings[0])

    qdrant_client.recreate_collection(
        collection_name=embedding_id,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=__get_distance_key(embedding_item.platform, embedding_item.model),
        ),
    )
    records = None

    if (
        embedding.get(project_id, embedding_id).platform
        == EmbeddingPlatform.PYTHON.value
    ):
        embeddings = [[float(e) for e in embedding] for embedding in embeddings]

    # extend payloads
    label_payload_extension = record_label_association.get_label_payload_for_qdrant(
        project_id
    )

    has_sub_key = embedding.has_sub_key(project_id, embedding_id)

    for record_id, payload in zip(record_ids, payloads):
        if record_id in label_payload_extension:
            payload[LABELS_QDRANT] = label_payload_extension[record_id]

    id_for_storage = None
    if has_sub_key:
        id_for_storage = tensor_ids
    else:
        id_for_storage = record_ids

    records = [
        models.Record(id=id, vector=e, payload=payload)
        for id, e, payload in zip(id_for_storage, embeddings, payloads)
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
    unlabeled_tensors = embedding.get_not_manually_labeled_tensors_by_embedding_id(
        project_id, embedding_id, 10000
    )
    if len(unlabeled_tensors) < 1:
        return status.HTTP_200_OK, [[], []]

    embedding_item = embedding.get(project_id, embedding_id)

    unlabeled_ids, unlabeled_embeddings = zip(*unlabeled_tensors)
    unlabeled_embeddings = np.array(unlabeled_embeddings)

    labeled_tensors = embedding.get_manually_labeled_tensors_by_embedding_id(
        project_id, embedding_id, 10000
    )

    if len(labeled_tensors) < 1:
        labeled_embeddings = np.mean(unlabeled_embeddings, axis=0, keepdims=True)
    else:
        _, labeled_embeddings = zip(*labeled_tensors)
        labeled_embeddings = np.array(labeled_embeddings)

    outlier_scores = np.sum(
        cdist(
            labeled_embeddings,
            unlabeled_embeddings,
            __get_distance_key(embedding_item.platform, embedding_item.model, False),
        ),
        axis=0,
    )
    sorted_index = np.argsort(
        outlier_scores,
        axis=None,
    )[::-1]

    count_unlabeled = record.count_records_without_manual_label(project_id)
    max_records = min(round(0.05 * count_unlabeled), limit)

    i = 0
    outlier_slice_ids = []
    outlier_slics_scores = []
    while len(outlier_slice_ids) < max_records and i < len(sorted_index):
        outlier_id = unlabeled_ids[sorted_index[i]]
        if outlier_id not in outlier_slice_ids:
            outlier_slice_ids.append(outlier_id)
            outlier_slics_scores.append(outlier_scores[sorted_index[i]])
        i += 1

    return status.HTTP_200_OK, [outlier_slice_ids, outlier_slics_scores]


def update_attribute_payloads(
    project_id: str,
    embedding_id: str,
    record_ids: Optional[List[str]],
) -> bool:
    if not __qdrant_collection_exits(embedding_id):
        raise ValueError(f"Collection {embedding_id} does not exist.")
    has_sub_key = embedding.has_sub_key(project_id, embedding_id)
    filter_attribute_dict = embedding.get_filter_attribute_type_dict(
        project_id, embedding_id
    )
    label_payload_extension = record_label_association.get_label_payload_for_qdrant(
        project_id,
        source_type=[LabelSource.MANUAL.value, LabelSource.WEAK_SUPERVISION.value],
        record_ids=record_ids,
    )

    if has_sub_key:
        all_object = embedding.get_tensors_and_attributes_for_qdrant(
            project_id, embedding_id, filter_attribute_dict, record_ids, True
        )
        record_ids, payloads, tensor_ids = zip(*all_object)
        ids_for_storage = tensor_ids
    else:
        all_object = embedding.get_attributes_for_qdrant(
            project_id, record_ids, filter_attribute_dict
        )
        record_ids, payloads = zip(*all_object)
        ids_for_storage = record_ids

    for record_id, payload in zip(record_ids, payloads):
        if record_id in label_payload_extension:
            payload[LABELS_QDRANT] = label_payload_extension[record_id]

    update_operations = [
        # use overwrite payload operation so that existing attributes in payload are
        # removed if not present in new payload but therefore we need to add the labels
        models.OverwritePayloadOperation(
            overwrite_payload=models.SetPayload(
                payload=payload,
                points=[point_id],
            )
        )
        for point_id, payload in zip(ids_for_storage, payloads)
    ]

    qdrant_client.batch_update_points(
        collection_name=embedding_id,
        update_operations=update_operations,
    )


def update_label_payloads(
    project_id: str, embedding_ids: List[str], record_ids: Optional[List[str]] = None
) -> None:
    label_payload_extension = record_label_association.get_label_payload_for_qdrant(
        project_id,
        source_type=[LabelSource.MANUAL.value, LabelSource.WEAK_SUPERVISION.value],
        record_ids=record_ids,
    )

    for embedding_id in embedding_ids:
        has_sub_key = embedding.has_sub_key(project_id, embedding_id)
        if has_sub_key:
            tensor_ids, record_ids = zip(
                *embedding.get_tensor_ids_and_record_ids_by_embedding_id(
                    embedding_id, record_ids
                )
            )
            ids_for_storage = tensor_ids
        else:
            if record_ids is None:
                record_ids = record.get_all_ids(project_id)
                ids_for_storage = record_ids
            else:
                ids_for_storage = record_ids

        payloads = []
        for record_id in record_ids:
            if record_id in label_payload_extension:
                payloads.append({LABELS_QDRANT: label_payload_extension[record_id]})
            else:
                payloads.append(None)

        update_operations = []
        for point_id, payload in zip(ids_for_storage, payloads):
            if payload is not None:
                update_operations.append(
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload=payload,
                            points=[point_id],
                        )
                    )
                )
            else:
                update_operations.append(
                    models.DeletePayloadOperation(
                        delete_payload=models.DeletePayload(
                            keys=[LABELS_QDRANT],
                            points=[point_id],
                        )
                    )
                )

        qdrant_client.batch_update_points(
            collection_name=embedding_id,
            update_operations=update_operations,
        )


def collection_exists(
    project_id: str, embedding_id: str, include_db_check: bool
) -> bool:
    if not __qdrant_collection_exits(embedding_id):
        return False
    if include_db_check and not embedding.get(project_id, embedding_id):
        return False
    return True


def __qdrant_collection_exits(collection_name: str) -> bool:
    # to be replaced by qdrant_client.collection_exists(collection_name=collection_name) after qdrant_client update
    try:
        qdrant_client.get_collection(collection_name)
        return True
    except Exception:
        return False


def __get_distance_key(
    platform: str, model: str, for_qdrant: bool = True
) -> Union[str, models.Distance]:
    if for_qdrant:
        if platform == EmbeddingPlatform.PYTHON.value and model == "tf-idf":
            if for_qdrant:
                return models.Distance.COSINE
            else:
                return "cosine"
        if for_qdrant:
            return models.Distance.EUCLID
        else:
            return "euclidean"
