from llm_accel.metrics.environment import collect_environment_metadata


def test_collect_environment_metadata_includes_reproducibility_fields() -> None:
    metadata = collect_environment_metadata(hardware_label="ci-smoke")

    assert metadata["python_version"]
    assert metadata["operating_system"]
    assert metadata["hardware_label"] == "ci-smoke"
    assert "git_commit" in metadata
    assert "gpu_name" in metadata
