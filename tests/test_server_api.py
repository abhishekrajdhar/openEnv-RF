from fastapi.testclient import TestClient

from support_queue_env.server.app import app


client = TestClient(app)


def test_health_and_metadata_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}

    metadata = client.get("/metadata")
    assert metadata.status_code == 200
    assert metadata.json()["name"] == "support_queue_env"


def test_reset_accepts_alias_and_step_requires_episode():
    step_before_reset = client.post(
        "/step",
        json={"action_type": "set_priority", "argument": "high"},
    )
    assert step_before_reset.status_code == 400
    assert "Call reset() before step()." in step_before_reset.json()["detail"]

    reset = client.post("/reset", json={"task_id": "01"})
    assert reset.status_code == 200
    assert reset.json()["observation"]["task"]["task_id"] == "delayed_shipping_refund"
