import json
from types import SimpleNamespace

import pandas as pd
import pytest
from openai import OpenAIError

from football_value import scouting
from football_value.features import TARGET

COLUMNS = [
    "name",
    "team",
    "competition",
    "position",
    "goals",
    "assists",
    "minutes_played",
    "played_matches",
    "age",
    "table_position",
    "points",
    "cl_stage",
    TARGET,
    "predicted_value_eur",
]


@pytest.fixture
def predictions():
    rows = [
        # Model higher than the market, 1.8 times: within the normal margin.
        ("Vedat Muriqi", "RCD Mallorca", "PD", "Attack", 23, 1, 3137, 37, 32.18, 17, 42, 0,
         4_500_000, 8_100_000),
        # Model lower than the market, 2 times, expensive player.
        ("Kylian Mbappé", "Real Madrid CF", "PD", "Attack", 25, 4, 2606, 31, 27.53, 2, 83, 5,
         180_000_000, 90_000_000),
        # Model 6 times higher, cheap defender with few minutes.
        ("Nobel Mendy", "Rayo Vallecano", "PD", "Defence", 2, 0, 250, 6, 21.9, 12, 50, 0,
         1_000_000, 6_000_000),
        # Model and market agree.
        ("Batista Mendy", "Sevilla FC", "PD", "Midfield", 1, 1, 1500, 25, 26.4, 14, 47, 0,
         8_000_000, 7_000_000),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["predicted_to_actual_ratio"] = frame["predicted_value_eur"] / frame[TARGET]
    return frame


@pytest.fixture
def metrics():
    return pd.DataFrame(
        [
            {"model": "baseline", "median_pct_error": 73.2, "cv_median_pct_error": 72.6},
            {"model": "ridge", "median_pct_error": 33.0, "cv_median_pct_error": 32.8},
        ]
    )


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


def facts_of(name, predictions, typical_error=33.0):
    player = scouting.find_player(predictions, name)
    return scouting.player_facts(player, predictions, typical_error)


# --- Finding the player -------------------------------------------------------


def test_find_player_ignores_accents_and_case(predictions):
    assert scouting.find_player(predictions, "kylian mbappe")["name"] == "Kylian Mbappé"


def test_find_player_accepts_part_of_the_name(predictions):
    assert scouting.find_player(predictions, "Muriqi")["name"] == "Vedat Muriqi"


def test_find_player_reports_unknown_and_ambiguous_names(predictions):
    with pytest.raises(scouting.PlayerNotFoundError, match="No player"):
        scouting.find_player(predictions, "Zidane")
    with pytest.raises(scouting.PlayerNotFoundError, match="Batista Mendy, Nobel Mendy"):
        scouting.find_player(predictions, "Mendy")


# --- Facts and limitations ----------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(84_511_000, "€84.5M"), (75_000_000, "€75M"), (4_500_000, "€4.5M"), (300_000, "€300K")],
)
def test_format_euros(amount, expected):
    assert scouting.format_euros(amount) == expected


def test_a_small_gap_is_described_as_agreement_with_the_typical_error():
    text = scouting.describe_gap(1.3, typical_error_percent=33)

    assert "broadly agree" in text
    assert "33%" in text


def test_a_large_gap_is_described_in_both_directions():
    assert "3.0 times higher" in scouting.describe_gap(3.0, 33)
    assert "4.0 times lower" in scouting.describe_gap(0.25, 33)


def test_no_limitation_is_given_when_model_and_market_agree(predictions):
    for name in ("Muriqi", "Batista Mendy"):
        assert scouting.relevant_limitations(scouting.find_player(predictions, name)) == []


def test_limitations_depend_on_the_player(predictions):
    expensive = scouting.relevant_limitations(scouting.find_player(predictions, "Mbappé"))
    cheap_defender = scouting.relevant_limitations(scouting.find_player(predictions, "Nobel"))

    assert expensive == [scouting.LIMITATION_EXPENSIVE_PLAYER]
    assert cheap_defender == [
        scouting.LIMITATION_DEFENDING,
        scouting.LIMITATION_CHEAP_PLAYER,
        scouting.LIMITATION_FEW_MINUTES,
    ]


def test_player_facts_contain_the_numbers_and_the_club_context(predictions):
    facts = facts_of("Mbappé", predictions)

    assert facts["player"]["league"] == "La Liga"
    assert facts["season_statistics"]["goals"] == 25
    assert facts["season_statistics"]["league_matches_played"] == 31
    assert facts["club_context"] == {
        "league_position": 2,
        "points": 83,
        "champions_league": "semi-finals",
    }
    assert facts["valuation"]["market_value"] == "€180M"
    assert facts["valuation"]["model_prediction"] == "€90M"
    assert facts["valuation"]["prediction_divided_by_market_value"] == 0.5
    assert "2.0 times lower" in facts["valuation"]["assessment"]
    assert facts["known_limitations_that_apply"] == [scouting.LIMITATION_EXPENSIVE_PLAYER]


def test_percentiles_compare_with_players_of_the_same_position(predictions):
    facts = facts_of("Muriqi", predictions)

    # Two attackers: Muriqi has fewer goals than Mbappé but more minutes.
    assert facts["percentile_among_same_position"]["goals"] == 50
    assert facts["percentile_among_same_position"]["minutes_played"] == 100


def test_messages_embed_the_facts_as_json(predictions):
    facts = facts_of("Muriqi", predictions)

    system, user = scouting.build_messages(facts, scouting.PLAYER_INSTRUCTIONS)

    assert system == {"role": "system", "content": scouting.SYSTEM_PROMPT}
    assert user["content"].startswith(scouting.PLAYER_INSTRUCTIONS)
    assert json.loads(user["content"].split("DATA:\n", 1)[1]) == facts


# --- Calling the LLM ----------------------------------------------------------


def test_main_model_error_reads_the_cross_validated_error(metrics):
    assert scouting.main_model_error(metrics) == 32.8


def test_player_report_sends_the_configured_model_and_returns_the_answer(predictions, metrics):
    client = FakeClient(answer="  Muriqi had a great season.  ")

    report = scouting.write_player_report(
        "Muriqi", predictions, metrics, client, model="some-model"
    )

    assert report == "Muriqi had a great season."
    assert client.requests[0]["model"] == "some-model"
    sent = client.requests[0]["messages"][1]["content"]
    assert "Vedat Muriqi" in sent
    assert "about 33%" in sent  # the typical error comes from the metrics file


def test_model_summary_sends_metrics_top_coefficients_and_limitations():
    metrics = pd.DataFrame([{"model": "ridge", "median_pct_error": 33.0, "r2_log": 0.86}])
    coefficients = pd.DataFrame(
        [{"feature": f"factor_{i}", "coefficient": 0.1, "effect_percent": 10.5} for i in range(20)]
    )
    client = FakeClient()

    scouting.write_model_summary(metrics, coefficients, client, model="some-model")

    sent = json.loads(client.requests[0]["messages"][1]["content"].split("DATA:\n", 1)[1])
    assert sent["metrics_by_model"]["values"][0]["r2_log"] == 0.86
    assert len(sent["largest_coefficients_of_the_main_model"]["values"]) == 12
    assert sent["known_limitations"] == scouting.MODEL_LIMITATIONS


def test_provider_errors_and_empty_answers_become_llm_errors():
    messages = [{"role": "user", "content": "hello"}]

    with pytest.raises(scouting.LLMError, match="model is paused"):
        scouting.generate_text(FakeClient(error=OpenAIError("model is paused")), "m", messages)
    with pytest.raises(scouting.LLMError, match="empty"):
        scouting.generate_text(FakeClient(answer="   "), "m", messages)


def test_command_line_exits_cleanly_when_the_key_is_missing(monkeypatch, caplog):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        scouting.config,
        "load_settings",
        lambda: scouting.config.Settings(None, None, "http://invalid", "some-model"),
    )

    exit_code = scouting.main(["model"])

    assert exit_code == 1
    assert "LLM_API_KEY is not set" in caplog.text
