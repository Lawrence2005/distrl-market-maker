"""
tests/test_process_lobster.py

Tests for data/process_lobster.py's process_lobster_directory error handling.
(compute_agent_params and its tests were archived alongside the rest of the
background-agent calibration pipeline — see
archive/lobster_calibration/test_compute_agent_params.py.)

Run with:
    python -m pytest tests/test_process_lobster.py -v
"""

import pytest


class TestProcessLobsterDirectory:
    def test_empty_dir_raises_file_not_found(self, tmp_path):
        from data.process_lobster import process_lobster_directory
        with pytest.raises(FileNotFoundError, match="No message files found"):
            process_lobster_directory(
                data_dir=str(tmp_path),
                n_levels=10,
                output_snapshots=str(tmp_path / "snaps.npy"),
            )

    def test_nonexistent_dir_raises(self, tmp_path):
        from data.process_lobster import process_lobster_directory
        with pytest.raises((FileNotFoundError, OSError)):
            process_lobster_directory(
                data_dir=str(tmp_path / "does_not_exist"),
                n_levels=10,
                output_snapshots=str(tmp_path / "snaps.npy"),
            )
