"""Tests for data format operations in transcribe_audio module."""

import json

import duckdb
import pandas as pd
import pytest

from src.transcribe_audio import (
    create_transcription_record,
    load_existing_results,
    save_results,
    save_to_csv,
    save_to_duckdb,
    save_to_json,
    save_to_parquet,
)


class TestTranscriptionRecordCreation:
    """Test transcription record creation."""

    def test_create_transcription_record(self):
        """Test creating a transcription record with all fields."""
        from datetime import UTC, datetime

        started_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        record = create_transcription_record(
            filename="test.mp3",
            file_id="abc123",
            file_size=1024,
            transcription_time=2.5,
            transcription_text="Hello world",
            model_id="openai/whisper-small",
            started_at=started_at,
        )

        expected_fields = {
            "file_id",
            "filename",
            "file_size_bytes",
            "transcription_time_seconds",
            "transcription_text",
            "model_id",
            "started_at",
            "processed_at",
        }

        assert set(record.keys()) == expected_fields
        assert record["filename"] == "test.mp3"
        assert record["file_id"] == "abc123"
        assert record["file_size_bytes"] == 1024
        assert record["transcription_time_seconds"] == 2.5
        assert record["transcription_text"] == "Hello world"
        assert record["model_id"] == "openai/whisper-small"
        assert record["started_at"] == started_at.isoformat()
        assert "processed_at" in record


class TestCSVOperations:
    """Test CSV save/load operations."""

    def test_save_to_csv_new_file(self, temp_dir, sample_transcription_records):
        """Test saving records to a new CSV file."""
        csv_file = temp_dir / "test.csv"

        save_to_csv(sample_transcription_records, csv_file)

        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_save_to_csv_existing_file(self, temp_dir, sample_transcription_records):
        """Test appending records to existing CSV file."""
        csv_file = temp_dir / "test.csv"

        # Save initial records
        save_to_csv(sample_transcription_records[:1], csv_file)
        # Append more records
        save_to_csv(sample_transcription_records[1:], csv_file)

        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_load_existing_results_csv(self, temp_dir, sample_transcription_records):
        """Test loading existing results from CSV."""
        csv_file = temp_dir / "test.csv"
        save_to_csv(sample_transcription_records, csv_file)

        file_ids = load_existing_results("csv", csv_file)

        assert file_ids == {"abc123", "def456"}

    def test_load_existing_results_csv_missing_file(self, temp_dir):
        """Test loading from non-existent CSV file."""
        csv_file = temp_dir / "missing.csv"

        file_ids = load_existing_results("csv", csv_file)

        assert file_ids == set()


class TestJSONOperations:
    """Test JSON save/load operations."""

    def test_save_to_json_new_file(self, temp_dir, sample_transcription_records):
        """Test saving records to a new JSON file."""
        json_file = temp_dir / "test.json"

        save_to_json(sample_transcription_records, json_file)

        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["filename"] == "test1.mp3"

    def test_save_to_json_existing_file(self, temp_dir, sample_transcription_records):
        """Test appending records to existing JSON file."""
        json_file = temp_dir / "test.json"

        # Save initial records
        save_to_json(sample_transcription_records[:1], json_file)
        # Append more records
        save_to_json(sample_transcription_records[1:], json_file)

        with open(json_file) as f:
            data = json.load(f)
        assert len(data) == 2
        assert [record["filename"] for record in data] == ["test1.mp3", "test2.wav"]

    def test_load_existing_results_json(self, temp_dir, sample_transcription_records):
        """Test loading existing results from JSON."""
        json_file = temp_dir / "test.json"
        save_to_json(sample_transcription_records, json_file)

        file_ids = load_existing_results("json", json_file)

        assert file_ids == {"abc123", "def456"}

    def test_save_to_json_unicode_handling(self, temp_dir):
        """Test that JSON properly handles Unicode characters."""
        json_file = temp_dir / "unicode.json"
        records = [
            {
                "file_id": "test123",
                "filename": "test_file.mp3",
                "transcription_text": "Hello 世界! こんにちは",
                "file_size_bytes": 1024,
                "transcription_time_seconds": 1.0,
                "model_id": "test-model",
                "started_at": "2024-01-01T00:00:00Z",
                "processed_at": "2024-01-01T00:00:01Z",
            }
        ]

        save_to_json(records, json_file)

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["transcription_text"] == "Hello 世界! こんにちは"


class TestParquetOperations:
    """Test Parquet save/load operations."""

    def test_save_to_parquet_new_file(self, temp_dir, sample_transcription_records):
        """Test saving records to a new Parquet file."""
        parquet_file = temp_dir / "test.parquet"

        save_to_parquet(sample_transcription_records, parquet_file)

        assert parquet_file.exists()
        df = pd.read_parquet(parquet_file)
        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_save_to_parquet_existing_file(
        self, temp_dir, sample_transcription_records
    ):
        """Test appending records to existing Parquet file."""
        parquet_file = temp_dir / "test.parquet"

        # Save initial records
        save_to_parquet(sample_transcription_records[:1], parquet_file)
        # Append more records
        save_to_parquet(sample_transcription_records[1:], parquet_file)

        df = pd.read_parquet(parquet_file)
        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_load_existing_results_parquet(
        self, temp_dir, sample_transcription_records
    ):
        """Test loading existing results from Parquet."""
        parquet_file = temp_dir / "test.parquet"
        save_to_parquet(sample_transcription_records, parquet_file)

        file_ids = load_existing_results("parquet", parquet_file)

        assert file_ids == {"abc123", "def456"}


class TestDuckDBOperations:
    """Test DuckDB save/load operations."""

    def test_save_to_duckdb_new_file(self, temp_dir, sample_transcription_records):
        """Test saving records to a new DuckDB file."""
        db_file = temp_dir / "test.duckdb"

        save_to_duckdb(sample_transcription_records, db_file)

        assert db_file.exists()
        conn = duckdb.connect(str(db_file))
        df = conn.execute("SELECT * FROM transcriptions").fetchdf()
        conn.close()

        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_save_to_duckdb_existing_file(self, temp_dir, sample_transcription_records):
        """Test appending records to existing DuckDB file."""
        db_file = temp_dir / "test.duckdb"

        # Save initial records
        save_to_duckdb(sample_transcription_records[:1], db_file)
        # Append more records
        save_to_duckdb(sample_transcription_records[1:], db_file)

        conn = duckdb.connect(str(db_file))
        df = conn.execute("SELECT * FROM transcriptions ORDER BY filename").fetchdf()
        conn.close()

        assert len(df) == 2
        assert list(df["filename"]) == ["test1.mp3", "test2.wav"]

    def test_save_to_duckdb_duplicate_handling(
        self, temp_dir, sample_transcription_records
    ):
        """Test that DuckDB handles duplicate file_ids correctly."""
        db_file = temp_dir / "test.duckdb"

        # Save records twice
        save_to_duckdb(sample_transcription_records, db_file)
        save_to_duckdb(sample_transcription_records, db_file)  # Should be ignored

        conn = duckdb.connect(str(db_file))
        df = conn.execute("SELECT * FROM transcriptions").fetchdf()
        conn.close()

        assert len(df) == 2  # Should still be 2, not 4

    def test_load_existing_results_duckdb(self, temp_dir, sample_transcription_records):
        """Test loading existing results from DuckDB."""
        db_file = temp_dir / "test.duckdb"
        save_to_duckdb(sample_transcription_records, db_file)

        file_ids = load_existing_results("duckdb", db_file)

        assert file_ids == {"abc123", "def456"}


class TestSaveResultsGeneral:
    """Test the general save_results function."""

    def test_save_results_all_formats(self, temp_dir, sample_transcription_records):
        """Test saving results in all supported formats."""
        formats = ["csv", "json", "parquet", "duckdb"]

        for format_name in formats:
            output_file = temp_dir / f"test.{format_name}"

            save_results(sample_transcription_records, format_name, output_file)

            assert output_file.exists()
            file_ids = load_existing_results(format_name, output_file)
            assert file_ids == {"abc123", "def456"}

    def test_save_results_empty_records(self, temp_dir):
        """Test saving empty records list."""
        output_file = temp_dir / "empty.csv"

        save_results([], "csv", output_file)

        # File should not be created for empty records
        assert not output_file.exists()

    def test_save_results_invalid_format(self, temp_dir, sample_transcription_records):
        """Test saving with invalid format raises error."""
        output_file = temp_dir / "test.invalid"

        with pytest.raises(ValueError, match="Unsupported output format"):
            save_results(sample_transcription_records, "invalid", output_file)


class TestLoadExistingResultsErrorHandling:
    """Test error handling in load_existing_results."""

    def test_load_existing_results_corrupted_csv(self, temp_dir):
        """Test loading from corrupted CSV file."""
        csv_file = temp_dir / "corrupted.csv"
        csv_file.write_text("invalid,csv,content\nwith,missing,data")

        file_ids = load_existing_results("csv", csv_file)

        assert file_ids == set()  # Should return empty set on error

    def test_load_existing_results_corrupted_json(self, temp_dir):
        """Test loading from corrupted JSON file."""
        json_file = temp_dir / "corrupted.json"
        json_file.write_text("{ invalid json content")

        file_ids = load_existing_results("json", json_file)

        assert file_ids == set()  # Should return empty set on error

    def test_load_existing_results_missing_file_id_column(self, temp_dir):
        """Test loading from file without file_id column."""
        csv_file = temp_dir / "no_file_id.csv"
        df = pd.DataFrame({"filename": ["test.mp3"], "content": ["some text"]})
        df.to_csv(csv_file, index=False)

        file_ids = load_existing_results("csv", csv_file)

        assert file_ids == set()  # Should return empty set if no file_id column
