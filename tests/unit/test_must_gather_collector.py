from pathlib import Path

from utilities.must_gather_collector import archive_must_gather, run_must_gather_clean


def test_archive_must_gather_writes_archive_under_target_directory(tmp_path: Path) -> None:
    source_dir = tmp_path / "must-gather"
    target_dir = tmp_path / "results"
    source_dir.mkdir()
    (source_dir / "report.yaml").write_text(data="report")

    archive_path = archive_must_gather(
        must_gather_path=str(source_dir),
        target_dir=str(target_dir),
        base_file_name="mg-123",
    )

    assert Path(archive_path) == target_dir / "mg-123.zip"
    assert Path(archive_path).is_file()
    assert not source_dir.exists()


def test_run_must_gather_clean_preserves_raw_data_when_cleaning_fails(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "report.yaml").write_text(data="report")
    target_dir = tmp_path / "results"
    cleaned_dir = target_dir / "must-gather-cleaned"

    monkeypatch.setattr(
        "utilities.must_gather_collector.run_command",
        lambda **_: (1, "", "permission denied"),
    )

    result = run_must_gather_clean(must_gather_path=str(raw_dir), target_dir=str(target_dir))

    assert result == str(raw_dir)
    assert raw_dir.exists()
    assert not cleaned_dir.exists()
