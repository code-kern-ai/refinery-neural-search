# -*- coding: utf-8 -*-
import os
import numpy as np
from fastapi import FastAPI, responses, status
from qdrant_client import QdrantClient
from neural_search.similarity_threshold import SimilarityThreshold
from scipy.spatial.distance import cdist

from submodules.model.business_objects import embedding

app = FastAPI()
port = int(os.environ["QDRANT_PORT"])
qdrant_client = QdrantClient(host="qdrant", port=port)

sim_thr = SimilarityThreshold(qdrant_client)

missing_collections_creation_in_progress = False


@app.get("/most_similar")
def most_similar(
    project_id: str, embedding_id: str, record_id: str, limit: int = 100
) -> responses.JSONResponse:
    """Find the n most similar records with respect to the specified record.

    Args:
        embedding_id (str): Embedding id.
        record_id (str): The record for which similar records are searched.
        limit (int): Specifies the maximum amount of returned records.
    Returns:
        JSONResponse: containing HTML status code and the n most similar records
    """
    embedding_item = embedding.get_tensor(embedding_id, record_id)
    query_vector = np.array(embedding_item.data)
    search_result = qdrant_client.search(
        collection_name=embedding_id,
        query_vector=query_vector,
        query_filter=None,
        top=limit,
    )

    # only return records which are more similar than the threshold
    similarity_threshold = sim_thr.get_threshold(project_id, embedding_id)
    similar_records = [
        result.id for result in search_result if result.score >= similarity_threshold
    ]

    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[similar_records, ""],
    )


@app.post("/recreate_collection")
def recreate_collection(project_id: str, embedding_id: str) -> responses.JSONResponse:
    """Create collection in Qdrant for the given embedding.

    Args:
        embedding_id (str): Embedding id.
    Returns:
        JSONResponse: html status code
    """
    tensors = embedding.get_tensors_by_embedding_id(embedding_id)
    ids, embeddings = zip(*tensors)
    embeddings = np.array(embeddings)

    if len(embeddings) == 0:
        return responses.JSONResponse(status_code=status.HTTP_404_NOT_FOUND)

    vector_size = embeddings.shape[-1]

    qdrant_client.recreate_collection(
        collection_name=embedding_id, vector_size=vector_size, distance="Euclid"
    )
    qdrant_client.upload_collection(
        collection_name=embedding_id, vectors=embeddings, payload=None, ids=ids
    )

    sim_thr.calculate_threshold(project_id, embedding_id)

    return responses.JSONResponse(status_code=status.HTTP_200_OK)


@app.get("/collections")
def get_collections() -> responses.JSONResponse:
    """
    Get list of existing collections.

    Returns:
        JSONResponse: html status code, list of collection names
    """
    response = qdrant_client.openapi_client.collections_api.get_collections()
    collections = [collection.name for collection in response.result.collections]
    return responses.JSONResponse(status_code=status.HTTP_200_OK, content=collections)


@app.put("/create_missing_collections")
def create_missing_collections() -> responses.JSONResponse:
    """
    Looks up embeddings for which no collection in Qdrant exists and creates these missing collections.

    Returns:
        JSONResponse: html status code
    """
    global missing_collections_creation_in_progress

    if missing_collections_creation_in_progress:
        return responses.JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content="The missing colletions are already being created.",
        )
    missing_collections_creation_in_progress = True

    response = qdrant_client.openapi_client.collections_api.get_collections()
    collections = [collection.name for collection in response.result.collections]
    embedding_items = embedding.get_finished_attribute_embeddings()

    if not embedding_items:
        missing_collections_creation_in_progress = False
        return responses.JSONResponse(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            content="There are no embeddings.",
        )

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

    return responses.JSONResponse(
        status_code=status.HTTP_200_OK, content=created_collections
    )


@app.put("/delete_collection")
def delete_collection(embedding_id: str) -> responses.JSONResponse:
    """
    Delete collection in Qdrant for the given embedding.

    Args:
        embedding_id (str)
    Returns:
        JSONResponse: html status code
    """
    qdrant_client.delete_collection(embedding_id)
    return responses.JSONResponse(status_code=status.HTTP_200_OK)


@app.get("/detect_outliers")
def detect_outliers(
    project_id: str, embedding_id: str, limit: int = 100
) -> responses.JSONResponse:
    """Detect outliers in the unlabeled records with respect to the already labeled records.
    The unlabeled record ids are returned sorted, beginning with the most outlying.

    Args:
        project_id (str): Project id.
        embedding_id (str): Embedding id.
        limit (int): Specifies the maximum amount of returned records.
    Returns:
        JSONResponse: html status code, if successfull the response the top_n most outlying records.
    """
    labeled_tensors = embedding.get_manually_labeled_tensors_by_embedding_id(
        project_id, embedding_id
    )
    labeled_ids, labeled_embeddings = zip(*labeled_tensors)
    labeled_embeddings = np.array(labeled_embeddings)

    if len(labeled_ids) < 1:
        return responses.JSONResponse(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            content="At least one record must be labeled manually to create outlier slice.",
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

    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[[outlier_ids.tolist(), outlier_scores.tolist()], ""],
    )
