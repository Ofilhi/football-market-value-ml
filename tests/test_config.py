import pytest

from football_value import config

ENV_VARS = ("FOOTBALL_DATA_API_KEY", "RESPAN_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Make every test start without any of the project's variables set."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "football-key")
    monkeypatch.setenv("RESPAN_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_MODEL", "some-model")

    settings = config.load_settings(env_file=None)

    assert settings.require_football_data_api_key() == "football-key"
    assert settings.require_llm_api_key() == "llm-key"
    assert settings.llm_model == "some-model"


def test_llm_defaults_apply_when_not_overridden():
    settings = config.load_settings(env_file=None)

    assert settings.llm_base_url == config.DEFAULT_LLM_BASE_URL
    assert settings.llm_model == config.DEFAULT_LLM_MODEL


def test_missing_key_raises_a_helpful_error():
    settings = config.load_settings(env_file=None)

    with pytest.raises(config.MissingSettingError, match="FOOTBALL_DATA_API_KEY"):
        settings.require_football_data_api_key()


def test_empty_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("RESPAN_API_KEY", "")
    settings = config.load_settings(env_file=None)

    with pytest.raises(config.MissingSettingError, match="RESPAN_API_KEY"):
        settings.require_llm_api_key()


def test_settings_are_read_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FOOTBALL_DATA_API_KEY=from-file\n")

    settings = config.load_settings(env_file=env_file)

    assert settings.football_data_api_key == "from-file"
