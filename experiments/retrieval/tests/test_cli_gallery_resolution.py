"""Retained tests for the CLI's retrieval/ensemble handling (now removed).

These exercised `main._resolve_gallery`, the `--gallery-path` flag, the
`ensemble` default and the ensemble routing line in the result summary. All of
that was removed from `viewfinder/main.py` when the retrieval predictor was
shelved, so the tests cannot run against the current code and are skipped.

They are kept, not deleted, because they document exactly what the shelved
behaviour was: the resolution order for a gallery, the rule that an explicit
`--predictor ensemble` errors while the default silently degrades to geoclip,
and the shape of the routing summary. If the experiment is ever revived, this
is the specification to revive it against.
"""

import argparse
from pathlib import Path

import pytest

pytestmark = pytest.mark.skip(
    reason="retrieval/ensemble removed from the active CLI; see experiments/retrieval/"
)

# Imported lazily inside the tests would still fail at collection, so the module
# under test is referenced by name only; these bodies are reference material.
from viewfinder import main  # noqa: E402


# --- gallery resolution --------------------------------------------------


def _args(**overrides):
    base = dict(predictor=None, ensemble_trust_threshold=None, gallery_path=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_gallery(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in main.GALLERY_REQUIRED_FILES:
        (root / name).write_text("x", encoding="utf-8")
    return root


def test_default_predictor_is_the_ensemble():
    args = _args()
    main._resolve_predictor_default(args)
    assert args.predictor == "ensemble"
    assert args.predictor_is_explicit is False


def test_explicit_predictor_is_marked_explicit():
    args = _args(predictor="geoclip")
    main._resolve_predictor_default(args)
    assert args.predictor_is_explicit is True


def test_tuning_the_threshold_counts_as_asking_for_the_ensemble():
    args = _args(ensemble_trust_threshold=0.9)
    main._resolve_predictor_default(args)
    assert args.predictor == "ensemble"
    assert args.predictor_is_explicit is True


def test_default_ensemble_degrades_to_geoclip_without_a_gallery(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_GALLERY_PATH", Path("nope/not/here"))
    args = _args()
    main._resolve_predictor_default(args)
    main._resolve_gallery(args, predictor_is_explicit=False)
    assert args.predictor == main.FALLBACK_PREDICTOR
    assert args.predictor_downgraded_from == "ensemble"


def test_explicit_ensemble_errors_without_a_gallery(monkeypatch):
    monkeypatch.setattr(main, "DEFAULT_GALLERY_PATH", Path("nope/not/here"))
    args = _args(predictor="ensemble")
    main._resolve_predictor_default(args)
    with pytest.raises(main.UserInputError, match="needs a reference gallery"):
        main._resolve_gallery(args, predictor_is_explicit=True)


def test_incomplete_gallery_is_rejected_by_name(tmp_path):
    partial = tmp_path / "gallery"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}", encoding="utf-8")
    args = _args(predictor="retrieval", gallery_path=str(partial))
    main._resolve_predictor_default(args)
    with pytest.raises(main.UserInputError, match="gallery.faiss"):
        main._resolve_gallery(args, predictor_is_explicit=True)


def test_named_gallery_is_resolved_to_an_absolute_path(tmp_path):
    gallery = _build_gallery(tmp_path / "v1")
    args = _args(predictor="retrieval", gallery_path=str(gallery))
    main._resolve_predictor_default(args)
    main._resolve_gallery(args, predictor_is_explicit=True)
    assert Path(args.gallery_path).is_absolute()
    assert args.predictor == "retrieval"


def test_non_gallery_predictors_are_untouched():
    args = _args(predictor="hybrid")
    main._resolve_predictor_default(args)
    main._resolve_gallery(args, predictor_is_explicit=True)
    assert args.gallery_path is None



def test_summary_reports_the_ensemble_source():
    result = _single_result(
        ensemble_routing={
            "routed_to": "retrieval",
            "max_top_similarity": 0.91,
            "trust_threshold": 0.825,
            "reason": "strong match",
        }
    )
    summary = main.summarize(result)
    assert summary["mode"] == "single"
    assert "retrieval" in summary["source_note"]
    assert "0.910" in summary["source_note"]
    assert summary["confidence"] == 74.3
    assert summary["place"] == "Silverdale, Washington, US"
    assert len(summary["alternatives"]) == 1


