from unittest.mock import patch

from fastapi.testclient import TestClient


def test_get_collections(client: TestClient):
    response = client.get("/collections")
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, list)


def test_healthcheck(client: TestClient):
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "OK"


def test_failed_healthcheck(client: TestClient):

    with patch(
        "submodules.model.business_objects.general.test_database_connection"
    ) as mock_execute:
        mock_execute.return_value = {
            "success": False,
            "error": "OperationalError",
        }
        response = client.get("/healthcheck")

    assert response.status_code == 500
    assert response.text == "database_error:OperationalError:"
