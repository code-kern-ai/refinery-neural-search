# -*- coding: utf-8 -*-
from fastapi import FastAPI, responses, status
from neural_search import util

app = FastAPI()


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
    similar_records = util.most_similar(project_id, embedding_id, record_id, limit)

    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[similar_records, ""],
    )


@app.post("/recreate_collection")
def recreate_collection(project_id: str, embedding_id: str) -> responses.HTMLResponse:
    """Create collection in Qdrant for the given embedding.

    Args:
        embedding_id (str): Embedding id.
    Returns:
        JSONResponse: html status code
    """
    status_code = util.recreate_collection(project_id, embedding_id)
    if status_code == 404:
        return responses.JSONResponse(status_code=status.HTTP_404_NOT_FOUND)
    return responses.HTMLResponse(status_code=status.HTTP_200_OK)


@app.get("/collections")
def get_collections() -> responses.JSONResponse:
    """
    Get list of existing collections.

    Returns:
        JSONResponse: html status code, list of collection names
    """
    collections = util.get_collections()
    return responses.JSONResponse(status_code=status.HTTP_200_OK, content=collections)


@app.put("/create_missing_collections")
def create_missing_collections() -> responses.JSONResponse:
    """
    Looks up embeddings for which no collection in Qdrant exists and creates these missing collections.

    Returns:
        JSONResponse: html status code
    """
    status_code, content = util.create_missing_collections()
    if status_code == 429:
        return responses.JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=content
        )
    elif status_code == 412:
        return responses.JSONResponse(
            status_code=status.HTTP_412_PRECONDITION_FAILED, content=content
        )
    return responses.JSONResponse(status_code=status.HTTP_200_OK, content=content)


@app.put("/delete_collection")
def delete_collection(embedding_id: str) -> responses.HTMLResponse:
    """
    Delete collection in Qdrant for the given embedding.

    Args:
        embedding_id (str)
    Returns:
        JSONResponse: html status code
    """
    util.delete_collection(embedding_id)
    return responses.HTMLResponse(status_code=status.HTTP_200_OK)


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
    status_code, content = util.detect_outliers(project_id, embedding_id, limit)
    if status_code == 412:
        return responses.JSONResponse(
            status_code=status.HTTP_412_PRECONDITION_FAILED, content=content
        )
    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[content, ""],
    )
