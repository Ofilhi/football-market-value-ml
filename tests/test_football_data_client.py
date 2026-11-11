import pytest

from football_value.football_data_client import (
    FootballDataClient,
    FootballDataError,
    cache_filename,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._payload


class FakeSession:
    """Returns the queued responses in order and records every call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._responses.pop(0)


class FakeTime:
    """A clock that only moves when someone sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(tmp_path, responses, fake_time=None):
    fake_time = fake_time or FakeTime()
    session = FakeSession(responses)
    client = FootballDataClient(
        api_key="test-key",
        cache_dir=tmp_path / "cache",
        requests_per_minute=10,
        session=session,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )
    return client, session, fake_time


def test_cache_filename_is_readable_and_independent_of_param_order():
    first = cache_filename("competitions/PL/scorers", {"season": 2025, "limit": 1000})
    second = cache_filename("/competitions/PL/scorers", {"limit": 1000, "season": 2025})

    assert first == second == "competitions_PL_scorers__limit-1000_season-2025.json"


def test_get_sends_the_api_key_and_returns_the_payload(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(payload={"count": 1})])

    assert client.get("competitions/PL/scorers", {"season": 2025}) == {"count": 1}

    url, kwargs = session.calls[0]
    assert url == "https://api.football-data.org/v4/competitions/PL/scorers"
    assert kwargs["headers"] == {"X-Auth-Token": "test-key"}
    assert kwargs["params"] == {"season": 2025}


def test_second_identical_request_is_served_from_the_cache(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(payload={"count": 1})])

    first = client.get("competitions/PL/scorers", {"season": 2025})
    second = client.get("competitions/PL/scorers", {"season": 2025})

    assert first == second
    assert len(session.calls) == 1


def test_requests_are_spaced_to_respect_the_rate_limit(tmp_path):
    client, _, fake_time = make_client(tmp_path, [FakeResponse(), FakeResponse()])

    client.get("competitions/PL/scorers")
    client.get("competitions/SA/scorers")

    # 10 requests per minute means at least 6 seconds between two requests.
    assert fake_time.sleeps == [6.0]


def test_rate_limit_response_is_retried(tmp_path):
    responses = [FakeResponse(status_code=429), FakeResponse(payload={"count": 1})]
    client, session, fake_time = make_client(tmp_path, responses)

    assert client.get("competitions/PL/scorers") == {"count": 1}
    assert len(session.calls) == 2
    assert 60.0 in fake_time.sleeps


def test_api_error_is_raised_with_its_message_and_not_cached(tmp_path):
    responses = [FakeResponse(status_code=403, payload={"message": "restricted resource"})]
    client, _, _ = make_client(tmp_path, responses)

    with pytest.raises(FootballDataError, match="403.*restricted resource"):
        client.get("competitions/XX/scorers")

    assert not (tmp_path / "cache").exists()
