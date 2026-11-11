import json
from types import SimpleNamespace

import pandas as pd
import pytest
from openai import OpenAIError

from football_value import scouting
from football_value.features import TARGET


@pytest.fixture
def predictions():
    rows = [
        ("Vedat Muriqi", "RCD Mallorca", "PD", "Attack", 23, 1, 3137, 32.18, 4_500_000, 32_974_000),
        ("Kylian Mbappé", "Real Madrid CF", "PD", "Attack", 25, 4, 2606, 27.53, 180_000_000, 9e7),
        ("Nobel Mendy", "Rayo Vallecano", "PD", "Defence", 2, 0, 900, 21.9, 3_000_000, 6_000_000),
        ("Batista Mendy", "Sevilla FC", "PD", "Midfield", 1, 1, 1500, 26.4, 8_000_000, 7_000_000),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "name",
            "team",
            "competition",
            "position",
            "goals",
            "assists",
            "minutes_played",
            "age",
            TARGET,
            "predicted_value_eur",
        ],
    )
    frame["predicted_to_actual_ratio"] = frame["predicted_value_eur"] / frame[TARGET]
    return frame


class FakeClient:
    """Stands in for the OpenAI client: records the request, returns a fixed answer."""

    def __init__(self, answer="A fine report.", error=None):
        self._answer = answer
        self._error = error
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_find_player_ignores_accents_and_case(predictions):
    assert scouting.find_player(predictions, "kylian mbappe")["name"] == "Kylian Mbappé"


def test_find_player_accepts_part_of_the_name(predictions):
    assert scouting.find_player(predictions, "Muriqi")["name"] == "Vedat Muriqi"


def test_find_player_reports_unknown_and_ambiguous_names(predictions):
    with pytest.raises(scouting.PlayerNotFoundError, match="No player"):
        scouting.find_player(predictions, "Zidane")
    with pytest.raises(scouting.PlayerNotFoundError, match="Batista Mendy, Nobel Mendy"):
        scouting.find_player(predictions, "Mendy")


def test_player_facts_contain_the_numbers_and_the_limitations(predictions):
    facts = scouting.player_facts(scouting.find_player(predictions, "Muriqi"), predictions)

    assert facts["player"]["league"] == "La Liga"
    assert facts["season_statistics"] == {"goals": 23, "assists": 1, "minutes_played": 3137}
    assert facts["valuation_eur"]["market_value"] == 4_500_000
    assert facts["valuation_eur"]["model_prediction"] == 32_974_000
    assert facts["valuation_eur"]["prediction_divided_by_market_value"] == 7.33
    # Two attackers: Muriqi has fewer goals than Mbappé but more minutes.
    assert facts["percentile_among_same_position"]["goals"] == 50
    assert facts["percentile_among_same_position"]["minutes_played"] == 100
    assert facts["known_limitations"] == scouting.KNOWN_LIMITATIONS


def test_messages_embed_the_facts_as_json(predictions):
    facts = scouting.player_facts(scouting.find_player(predictions, "Muriqi"), predictions)

    system, user = scouting.build_messages(facts, scouting.PLAYER_INSTRUCTIONS)

    assert system == {"role": "system", "content": scouting.SYSTEM_PROMPT}
    assert user["content"].startswith(scouting.PLAYER_INSTRUCTIONS)
    assert json.loads(user["content"].split("DATA:\n", 1)[1]) == facts


def test_player_report_sends_the_configured_model_and_returns_the_answer(predictions):
    client = FakeClient(answer="  Muriqi had a great season.  ")

    report = scouting.write_player_report("Muriqi", predictions, client, model="some-model")

    assert report == "Muriqi had a great season."
    assert client.requests[0]["model"] == "some-model"
    assert "Vedat Muriqi" in client.requests[0]["messages"][1]["content"]


def test_model_summary_sends_metrics_and_coefficients():
    metrics = pd.DataFrame([{"model": "linear", "median_pct_error": 62.8, "r2_log": 0.49}])
    coefficients = pd.DataFrame(
        [
            {"feature": "age", "coefficient": -0.66, "one_std": 4.2, "effect_percent": -48.2},
            {
                "feature": "position_Defence",
                "coefficient": 0.24,
                "one_std": None,
                "effect_percent": 26.9,
            },
        ]
    )
    client = FakeClient()

    scouting.write_model_summary(metrics, coefficients, client, model="some-model")

    sent = json.loads(client.requests[0]["messages"][1]["content"].split("DATA:\n", 1)[1])
    assert sent["test_set_metrics"]["values"][0]["r2_log"] == 0.49
    assert sent["linear_model_coefficients"]["values"][0]["feature"] == "age"
    assert sent["linear_model_coefficients"]["values"][1]["one_std"] is None


def test_provider_errors_and_empty_answers_become_llm_errors():
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(scouting.LLMError, match="model is paused"):
        scouting.generate_text(FakeClient(error=OpenAIError("model is paused")), "m", messages)
    with pytest.raises(scouting.LLMError, match="empty"):
        scouting.generate_text(FakeClient(answer="   "), "m", messages)


def test_command_line_exits_cleanly_when_the_key_is_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("RESPAN_API_KEY", raising=False)
    monkeypatch.setattr(scouting.config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        scouting.config,
        "load_settings",
        lambda: scouting.config.Settings(None, None, "http://invalid", "some-model"),
    )

    exit_code = scouting.main(["model"])

    assert exit_code == 1
    assert "RESPAN_API_KEY is not set" in caplog.text
