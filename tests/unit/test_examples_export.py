from pathlib import Path

from llm_accel.cli import main
from llm_accel.examples import write_example_files


def test_exported_matrix_example_runs_outside_the_repository(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_example_files(tmp_path / "examples")

    code = main(
        [
            "bench",
            "matrix",
            "--config",
            str(tmp_path / "examples" / "optimization_matrix_mock.yaml"),
            "--output-dir",
            str(tmp_path / "matrix"),
        ]
    )

    assert code == 0
    assert (tmp_path / "matrix" / "matrix_state.json").exists()
