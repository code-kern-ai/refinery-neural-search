# -*- coding: utf-8 -*-
from fastapi import FastAPI, responses, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from neural_search import util
from submodules.model.business_objects import general

app = FastAPI()


@app.post("/most_similar")
def most_similar(
    project_id: str,
    embedding_id: str,
    record_id: str,
    limit: int = 100,
    att_filter: Optional[List[Dict[str, Any]]] = None,
    record_sub_key: Optional[int] = None,
) -> responses.JSONResponse:
    """Find the n most similar records with respect to the specified record.

    Args:
        embedding_id (str): Embedding id.
        record_id (str): The record for which similar records are searched.
        limit (int): Specifies the maximum amount of returned records.
        att_filter(Optional[Dict[str, Any]]]): Specifies the attribute filter for the search as dict objects.

        example_filter = [
            {"key": "name", "value": ["John", "Doe"]}, -> name IN ("John", "Doe")
            {"key": "age", "value": 42}, -> age = 42
            {"key": "age", "value": [35,40]}, -> age IN (35,40)
            {"key": "age", "value": [35,40], type:"between"} -> age BETWEEN 35 AND 40 (includes 35 and 40)
        ]
    Returns:
        JSONResponse: containing HTML status code and the n most similar records
    """
    session_token = general.get_ctx_token()
    similar_records = util.most_similar(
        project_id, embedding_id, record_id, limit, att_filter, record_sub_key
    )
    general.remove_and_refresh_session(session_token)
    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=similar_records,
    )


class MostSimilarByEmbeddingRequest(BaseModel):
    project_id: str
    embedding_id: str
    embedding_tensor: List[float]
    limit: int = 5
    att_filter: Optional[List[Dict[str, Any]]] = None
    threshold: Optional[Union[float, int]] = None


@app.post("/most_similar_by_embedding")
def most_similar_by_embedding(
    request: MostSimilarByEmbeddingRequest,
) -> responses.JSONResponse:
    """Find the n most similar records with respect to the specified embedding.
        Args:
        embedding_id (str): Embedding id.
        record_id (str): The record for which similar records are searched.
        limit (int): Specifies the maximum amount of returned records.
        att_filter(Optional[Dict[str, Any]]]): Specifies the attribute filter for the search as dict objects.
        threshold: Optional[float]: None = calculated DB threshold, -9999 = no threshold, specified = use value
        example_filter = [
            {"key": "name", "value": ["John", "Doe"]}, -> name IN ("John", "Doe")
            {"key": "age", "value": 42}, -> age = 42
            {"key": "age", "value": [35,40]}, -> age IN (35,40)
            {"key": "age", "value": [35,40], type:"between"} -> age BETWEEN 35 AND 40 (includes 35 and 40)
        ]
    Returns:
        JSONResponse: containing HTML status code and the n most similar records
    """
    session_token = general.get_ctx_token()
    if isinstance(request.threshold, int):
        request.threshold = float(request.threshold)
    similar_records = util.most_similar_by_embedding(
        request.project_id,
        request.embedding_id,
        request.embedding_tensor,
        request.limit,
        request.att_filter,
        request.threshold,
    )
    general.remove_and_refresh_session(session_token)
    return responses.JSONResponse(
        status_code=status.HTTP_200_OK,
        content=similar_records,
    )


@app.post("/recreate_collection")
def recreate_collection(
    project_id: str, embedding_id: str
) -> responses.PlainTextResponse:
    """Create collection in Qdrant for the given embedding.

    Args:
        embedding_id (str): Embedding id.
    Returns:
        JSONResponse: html status code
    """
    session_token = general.get_ctx_token()
    status_code = util.recreate_collection(project_id, embedding_id)
    general.remove_and_refresh_session(session_token)
    return responses.PlainTextResponse(status_code=status_code)


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
    session_token = general.get_ctx_token()
    status_code, content = util.create_missing_collections()
    general.remove_and_refresh_session(session_token)
    return responses.JSONResponse(status_code=status_code, content=content)


@app.put("/delete_collection")
def delete_collection(embedding_id: str) -> responses.PlainTextResponse:
    """
    Delete collection in Qdrant for the given embedding.

    Args:
        embedding_id (str)
    Returns:
        JSONResponse: html status code
    """
    util.delete_collection(embedding_id)
    return responses.PlainTextResponse(status_code=status.HTTP_200_OK)


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
    session_token = general.get_ctx_token()
    status_code, content = util.detect_outliers(project_id, embedding_id, limit)
    general.remove_and_refresh_session(session_token)
    return responses.JSONResponse(
        status_code=status_code,
        content=content,
    )


@app.get("/healthcheck")
def healthcheck() -> responses.PlainTextResponse:
    text = ""
    status_code = status.HTTP_200_OK
    database_test = general.test_database_connection()
    if not database_test.get("success"):
        error_name = database_test.get("error")
        text += f"database_error:{error_name}:"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if not text:
        text = "OK"
    return responses.PlainTextResponse(text, status_code=status_code)
