from llm_accel.serving.health import check_endpoint_health


def test_mock_endpoint_health_is_available() -> None:
    health = check_endpoint_health("mock://local")

    assert health["healthy"] is True
    assert health["error"] is None
