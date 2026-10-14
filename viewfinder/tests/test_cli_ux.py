"""Tests for the CLI usability layer: progress bars, predictor default, summaries.

The gallery-resolution and ensemble-routing tests that used to live here moved
to experiments/retrieval/tests/ with the predictor they covered.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import pytest

from viewfinder import main
from viewfinder.progress import TerminalProgressBar, report


class FakeTTY(io.StringIO):
    """A StringIO that claims to be a terminal, so bars actually render."""

    def isatty(self) -> bool:
        return True


# --- progress ------------------------------------------------------------


def test_bar_renders_only_on_a_tty():
    plain = io.StringIO()
    with TerminalProgressBar("x", stream=plain) as bar:
        bar.update(5, 10)
    assert plain.getvalue() == ""

    tty = FakeTTY()
    with TerminalProgressBar("x", stream=tty) as bar:
        bar.update(5, 10)
    assert "5/10" in tty.getvalue()
    assert " 50.0%" in tty.getvalue()


def test_bar_clamps_and_survives_a_zero_total():
    tty = FakeTTY()
    bar = TerminalProgressBar("x", stream=tty)
    bar.update(50, 10)          # done > total
    bar.update(-3, 10)          # negative
    bar.update(7, 0)            # unknown total
    bar.close()
    bar.close()                 # idempotent
    assert "10/10" in tty.getvalue()


def test_bar_erases_a_longer_previous_line():
    tty = FakeTTY()
    bar = TerminalProgressBar("L", stream=tty)
    bar.update(1000, 10000)
    bar.update(2, 3)
    bar.close()
    # The short line must be padded past the length of the long one it replaced.
    assert tty.getvalue().split("\r")[-1].rstrip("\n").endswith("   ")


def test_report_bridges_bool_and_callable():
    log = logging.getLogger("t")
    seen: list[tuple[int, int]] = []

    report(lambda done, total: seen.append((done, total)), 3, 9, logger=log, message="%d/%d")
    assert seen == [(3, 9)]

    seen.clear()
    report(False, 3, 9, logger=log, message="%d/%d")
    assert seen == []


def test_report_swallows_a_broken_callback():
    """A failing UI must never take the pipeline down with it."""

    def explode(done: int, total: int) -> None:
        raise RuntimeError("ui is gone")

    report(explode, 1, 2, logger=logging.getLogger("t"), message="%d/%d")


# --- summary -------------------------------------------------------------


def _single_result(**extra):
    result = {
        "predictor": "geoclip",
        "prediction": {
            "latitude": 47.5,
            "longitude": -122.7,
            "country": "US",
            "admin1": "Washington",
            "city": "Silverdale",
            "confidence_percent": 74.3,
        },
        "predictions": [
            {
                "rank": 1,
                "lat": 47.5,
                "lon": -122.7,
                "confidence": 74.3,
                "country": "US",
                "admin1": "Washington",
                "city": "Silverdale",
            },
            {
                "rank": 2,
                "lat": 37.9,
                "lon": -122.6,
                "confidence": 36.6,
                "country": "US",
                "admin1": "California",
                "city": "Mill Valley",
            },
        ],
    }
    result.update(extra)
    return result


def test_comparison_is_not_reported_as_an_abstention():
    """A multi-predictor run has no combined answer by design; not an abstention."""
    result = {
        "predictors": ["geoclip", "plonk-osv"],
        "results": {
            "geoclip": {"prediction": {
                "latitude": 37.9, "longitude": -122.6,
                "country": "US", "admin1": "California", "city": "Mill Valley",
                "confidence_percent": 87.2,
            }},
            "plonk-osv": {"prediction": {
                "latitude": 47.6, "longitude": -122.7,
                "country": "US", "admin1": "Washington", "city": "Silverdale",
                "confidence_percent": 83.2,
            }},
        },
    }
    summary = main.summarize(result)
    assert summary["mode"] == "comparison"
    assert summary["abstained"] is False
    assert [entry["predictor"] for entry in summary["per_predictor"]] == [
        "geoclip",
        "plonk-osv",
    ]


def test_abstention_keeps_the_rejected_hypothesis():
    result = {
        "predictor": "hybrid",
        "prediction": None,
        "visual_hypotheses": [
            {
                "rank": 1,
                "lat": 37.44,
                "lon": -121.87,
                "confidence": 12.0,
                "country": "US",
                "admin1": "California",
                "city": "Milpitas",
            }
        ],
    }
    summary = main.summarize(result)
    assert summary["mode"] == "abstained"
    assert summary["abstained"] is True
    assert summary["place"] == "Milpitas, California, US"
    assert summary["confidence"] is None


def test_summary_prints_at_every_verbosity():
    """The summary is unconditional; only its destination stream varies."""
    stream = io.StringIO()
    main._print_summary(_single_result(), stream)
    text = stream.getvalue()
    assert "RESULT: Silverdale, Washington, US" in text
    assert "74.3%" in text


def test_comparison_summary_lists_each_predictor():
    result = {
        "predictors": ["geoclip", "plonk-osv"],
        "results": {
            "geoclip": {"prediction": {
                "latitude": 37.9, "longitude": -122.6,
                "country": "US", "admin1": "California", "city": "Mill Valley",
                "confidence_percent": 87.2,
            }},
        },
    }
    stream = io.StringIO()
    main._print_summary(result, stream)
    text = stream.getvalue()
    assert "not combined" in text
    assert "Mill Valley" in text


# --- predictor default ---------------------------------------------------


def test_default_predictor_is_geoclip():
    """Retrieval and ensemble are shelved; geoclip is the default going forward."""
    args = argparse.Namespace(predictor=None)
    main._resolve_predictor_default(args)
    assert args.predictor == "geoclip"
    assert main.DEFAULT_PREDICTOR == "geoclip"
    assert args.predictor_is_explicit is False


def test_explicit_predictor_is_marked_explicit():
    args = argparse.Namespace(predictor="hybrid")
    main._resolve_predictor_default(args)
    assert args.predictor == "hybrid"
    assert args.predictor_is_explicit is True


def test_shelved_predictors_are_rejected_by_the_parser():
    """retrieval/ensemble/both must no longer be selectable."""
    parser = main.build_parser()
    for shelved in ("retrieval", "ensemble", "both"):
        with pytest.raises(SystemExit):
            parser.parse_args(["x.jpg", "--predictor", shelved])
