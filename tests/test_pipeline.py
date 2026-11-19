import pytest

from football_value import pipeline


def test_steps_run_in_dependency_order(monkeypatch):
    calls = []
    for step in pipeline.STEPS:
        name = step.__name__.rsplit(".", 1)[-1]
        monkeypatch.setattr(step, "main", lambda name=name: calls.append(name))

    pipeline.run()

    assert calls == ["collect", "teams", "merge", "features", "eda", "model", "visualize"]


def test_a_failing_step_stops_the_pipeline(monkeypatch):
    calls = []
    for step in pipeline.STEPS:
        name = step.__name__.rsplit(".", 1)[-1]
        monkeypatch.setattr(step, "main", lambda name=name: calls.append(name))

    def broken():
        raise RuntimeError("merge failed")

    monkeypatch.setattr(pipeline.merge, "main", broken)

    with pytest.raises(RuntimeError, match="merge failed"):
        pipeline.run()

    assert calls == ["collect", "teams"]  # nothing after the failing step ran
