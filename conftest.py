import pytest
from fastapi.testclient import TestClient
from typing import Iterator
from app import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
