import pytest

from calibrayn.config import ROOT, ConfigError, Settings, Tuning, load_tuning
from calibrayn.orchestrator.zpd import ZPDConfig


def settings(**changes):
    return Settings(_env_file=None, **changes)


class TestSettings:
    def test_the_defaults_are_safe_and_work_offline(self):
        chosen = settings()
        assert chosen.llm_provider == "mock"
        assert chosen.privacy_mode == "local"
        assert chosen.embedder == "hashing"
        assert chosen.index == "numpy"

    def test_the_database_lives_in_the_data_folder_by_default(self, tmp_path):
        assert settings(data_dir=tmp_path).db_path == tmp_path / "calibrayn.db"

    def test_a_database_path_overrides_the_default(self, tmp_path):
        chosen = settings(data_dir=tmp_path, database_path=tmp_path / "other.db")
        assert chosen.db_path == tmp_path / "other.db"

    def test_environment_variables_override_the_defaults(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("PRIVACY_MODE", "remote_allowed")
        chosen = settings()
        assert chosen.llm_provider == "ollama"
        assert chosen.privacy_mode == "remote_allowed"

    def test_unknown_environment_variables_are_ignored(self, monkeypatch):
        monkeypatch.setenv("SOMETHING_ELSE", "1")
        assert settings().llm_provider == "mock"

    def test_the_default_files_exist(self):
        chosen = settings()
        assert chosen.settings_file.is_file()
        assert (chosen.prompts_dir / "question.txt").is_file()
        assert chosen.settings_file.parent.parent == ROOT


class TestLoadTuning:
    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_tuning(tmp_path / "nothing.yaml")

    def test_invalid_yaml_is_reported(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("zpd: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_tuning(path)

    def test_an_empty_file_gives_the_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_tuning(path) == Tuning()

    def test_a_block_can_change_one_setting(self, tmp_path):
        path = tmp_path / "some.yaml"
        path.write_text("zpd:\n  window: 12\n", encoding="utf-8")
        loaded = load_tuning(path)
        assert loaded.zpd == ZPDConfig(window=12)
        assert loaded.sm2 == Tuning().sm2

    def test_a_block_that_is_not_a_list_of_settings_is_reported(self, tmp_path):
        path = tmp_path / "odd.yaml"
        path.write_text("zpd: 5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="name: value"):
            load_tuning(path)

    def test_an_unknown_setting_is_reported_with_its_block(self, tmp_path):
        path = tmp_path / "typo.yaml"
        path.write_text("zpd:\n  windw: 12\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="'zpd'.*not recognised"):
            load_tuning(path)

    def test_a_bad_value_is_reported_with_its_block(self, tmp_path):
        path = tmp_path / "bad_value.yaml"
        path.write_text("zpd:\n  window: 0\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="'zpd'.*bad value"):
            load_tuning(path)

    def test_other_files_blocks_are_ignored(self, tmp_path):
        path = tmp_path / "extra.yaml"
        path.write_text("unrelated:\n  a: 1\n", encoding="utf-8")
        assert load_tuning(path) == Tuning()
