import json
import os
import time
import pytest
import tempfile

from teller_helpers import write_teller_state, cleanup_old_tts_files, is_debug_mode


class TestWriteTellerState:
    def test_creates_file_with_all_fields(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        write_teller_state(state_file, "idle", "")
        with open(state_file) as f:
            data = json.load(f)
        assert "teller_state" in data
        assert "latest_audio" in data

    def test_sets_teller_state_correctly(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        for state in ("idle", "thinking", "speaking"):
            write_teller_state(state_file, state)
            with open(state_file) as f:
                assert json.load(f)["teller_state"] == state

    def test_sets_latest_audio(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        write_teller_state(state_file, "speaking", "response_123.mp3")
        with open(state_file) as f:
            assert json.load(f)["latest_audio"] == "response_123.mp3"

    def test_latest_audio_defaults_to_empty_string(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        write_teller_state(state_file, "idle")
        with open(state_file) as f:
            assert json.load(f)["latest_audio"] == ""

    def test_overwrites_previous_state(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        write_teller_state(state_file, "thinking", "old.mp3")
        write_teller_state(state_file, "speaking", "new.mp3")
        with open(state_file) as f:
            data = json.load(f)
        assert data["teller_state"] == "speaking"
        assert data["latest_audio"] == "new.mp3"

    def test_writes_valid_json(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        write_teller_state(state_file, "thinking")
        with open(state_file) as f:
            content = f.read()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)


class TestCleanupOldTtsFiles:
    def _make_mp3(self, directory, name, offset_seconds=0):
        path = os.path.join(directory, name)
        with open(path, "wb") as f:
            f.write(b"\x00" * 100)
        if offset_seconds:
            t = time.time() - offset_seconds
            os.utime(path, (t, t))
        return path

    def test_keeps_only_most_recent_files(self, tmp_path):
        d = str(tmp_path)
        self._make_mp3(d, "response_001.mp3", offset_seconds=30)
        self._make_mp3(d, "response_002.mp3", offset_seconds=20)
        self._make_mp3(d, "response_003.mp3", offset_seconds=10)
        self._make_mp3(d, "response_004.mp3", offset_seconds=0)
        cleanup_old_tts_files(d, keep=2)
        remaining = sorted(os.listdir(d))
        assert remaining == ["response_003.mp3", "response_004.mp3"]

    def test_does_not_delete_when_under_keep_limit(self, tmp_path):
        d = str(tmp_path)
        self._make_mp3(d, "response_001.mp3")
        self._make_mp3(d, "response_002.mp3")
        cleanup_old_tts_files(d, keep=3)
        assert len(os.listdir(d)) == 2

    def test_does_nothing_on_empty_directory(self, tmp_path):
        cleanup_old_tts_files(str(tmp_path), keep=3)
        assert os.listdir(str(tmp_path)) == []

    def test_does_not_delete_non_mp3_files(self, tmp_path):
        d = str(tmp_path)
        other = os.path.join(d, "state.json")
        with open(other, "w") as f:
            f.write("{}")
        for i in range(5):
            self._make_mp3(d, f"response_00{i}.mp3", offset_seconds=10 * i)
        cleanup_old_tts_files(d, keep=2)
        assert os.path.exists(other)

    def test_keeps_exact_number_when_at_limit(self, tmp_path):
        d = str(tmp_path)
        for i in range(3):
            self._make_mp3(d, f"response_00{i}.mp3", offset_seconds=10 * i)
        cleanup_old_tts_files(d, keep=3)
        assert len([f for f in os.listdir(d) if f.endswith(".mp3")]) == 3


class TestIsDebugMode:
    def test_returns_true_when_env_is_1(self, monkeypatch):
        monkeypatch.setenv("TELLER_DEBUG", "1")
        assert is_debug_mode() is True

    def test_returns_false_when_env_is_0(self, monkeypatch):
        monkeypatch.setenv("TELLER_DEBUG", "0")
        assert is_debug_mode() is False

    def test_returns_false_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("TELLER_DEBUG", raising=False)
        assert is_debug_mode() is False
