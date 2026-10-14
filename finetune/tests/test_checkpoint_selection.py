"""Tests for dual-metric checkpoint selection.

The commons01 divergence is the reference case: these assert that selecting on
the external metric picks a different, correct checkpoint, and that a missing
external measurement is skipped rather than counted against patience.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkpoint_selection import (  # noqa: E402
    CheckpointSelector,
    divergence_summary,
    format_dual_trajectory,
    legacy_selector,
)

# The measured commons01 trajectory: the two metrics move in opposite directions.
COMMONS01 = [
    {"epoch": 0, "step": 107, "mean_loss": 4.19991,
     "median_error_km": 324.04, "external_median_error_km": 1164.60, "external_sign_z": 1.74},
    {"epoch": 2, "step": 321, "mean_loss": 3.13966,
     "median_error_km": 320.72, "external_median_error_km": 1199.11, "external_sign_z": 1.34},
    {"epoch": 4, "step": 535, "mean_loss": 2.43570,
     "median_error_km": 314.32, "external_median_error_km": 1453.44, "external_sign_z": -1.04},
    {"epoch": 6, "step": 749, "mean_loss": 1.93610,
     "median_error_km": 326.88, "external_median_error_km": 1478.65, "external_sign_z": -0.90},
]


def _run(selector: CheckpointSelector, points=COMMONS01) -> list[str]:
    """Feed a trajectory through a selector; return the files it wrote."""
    written: list[str] = []

    def save(filename: str) -> str:
        written.append(f"{filename}@epoch{point['epoch']}")
        return filename

    for point in points:
        selector.offer(point, epoch=point["epoch"], step=point["step"], save=save)
    return written


def test_indist_selector_picks_the_epoch_that_broke_commons01():
    """Reproduces the bug: in-distribution selection lands on epoch 4."""
    selector = CheckpointSelector(name="indist", metric_key="median_error_km")
    _run(selector)
    assert selector.epoch == 4
    assert selector.value == pytest.approx(314.32)


def test_external_selector_picks_epoch_zero():
    """The fix: external selection lands on epoch 0, the real best."""
    selector = CheckpointSelector(name="external", metric_key="external_median_error_km")
    _run(selector)
    assert selector.epoch == 0
    assert selector.value == pytest.approx(1164.60)


def test_the_two_selectors_disagree_on_this_run():
    indist = CheckpointSelector(name="indist", metric_key="median_error_km")
    external = CheckpointSelector(name="external", metric_key="external_median_error_km")
    _run(indist)
    _run(external)
    assert indist.epoch != external.epoch, "this run is the reason selection became plural"


def test_selectors_write_separate_files():
    indist = CheckpointSelector(name="indist", metric_key="median_error_km")
    external = CheckpointSelector(name="external", metric_key="external_median_error_km")
    assert indist.filename == "location_encoder_best_indist.pth"
    assert external.filename == "location_encoder_best_external.pth"
    assert indist.filename != external.filename


def test_missing_metric_is_skipped_not_counted_as_worse():
    """A coarser probe cadence must not spend early-stopping patience."""
    selector = CheckpointSelector(name="external", metric_key="external_median_error_km")
    save = lambda name: name  # noqa: E731

    selector.offer({"external_median_error_km": 1000.0}, epoch=0, step=1, save=save)
    assert selector.stale_evals == 0

    # Two evaluations with no external measurement at all.
    selector.offer({"median_error_km": 300.0}, epoch=1, step=2, save=save)
    selector.offer({"median_error_km": 290.0}, epoch=2, step=3, save=save)
    assert selector.stale_evals == 0, "absent measurement is not a worse measurement"

    # A genuine non-improvement does count.
    selector.offer({"external_median_error_km": 1200.0}, epoch=3, step=4, save=save)
    assert selector.stale_evals == 1


def test_none_valued_metric_is_also_skipped():
    selector = CheckpointSelector(name="external", metric_key="external_median_error_km")
    selector.offer({"external_median_error_km": None}, epoch=0, step=1, save=lambda n: n)
    assert selector.epoch is None
    assert selector.stale_evals == 0


def test_higher_is_better_metric_inverts_comparison():
    """The sign-test z is not an error; selecting on it must maximise."""
    selector = CheckpointSelector(
        name="z", metric_key="external_sign_z", lower_is_better=False
    )
    _run(selector)
    assert selector.epoch == 0
    assert selector.value == pytest.approx(1.74)


def test_legacy_selector_preserves_old_behaviour():
    selector = legacy_selector()
    assert selector.metric_key == "median_error_km"
    assert selector.drives_early_stop is True
    _run(selector)
    assert selector.epoch == 4


def test_divergence_summary_names_both_epochs_and_the_penalty():
    text = divergence_summary(COMMONS01)
    assert "DIVERGE" in text
    assert "epoch 4" in text and "epoch 0" in text
    # 1453.44 - 1164.60 = 288.84 km worse externally.
    assert "+288.84" in text


def test_divergence_summary_reports_agreement_when_metrics_agree():
    agreeing = [
        {"epoch": 0, "median_error_km": 400.0, "external_median_error_km": 1400.0},
        {"epoch": 1, "median_error_km": 300.0, "external_median_error_km": 1200.0},
    ]
    text = divergence_summary(agreeing)
    assert "agree" in text.lower()
    assert "DIVERGE" not in text


def test_dual_trajectory_table_flags_disagreeing_steps():
    table = format_dual_trajectory(COMMONS01)
    assert "indist_km" in table and "extern_km" in table
    # Epoch 4 improved in-distribution while degrading externally.
    assert "NO" in table
    assert "I" in table and "E" in table


# --- dense mid-epoch probing --------------------------------------------


DENSE = [
    {"epoch": 0, "step": 25, "mean_loss": 5.1, "mid_epoch": True,
     "median_error_km": 360.0, "external_median_error_km": 1200.0, "external_sign_z": 1.9},
    {"epoch": 0, "step": 50, "mean_loss": 4.6, "mid_epoch": True,
     "median_error_km": 340.0, "external_median_error_km": 1150.0, "external_sign_z": 2.1},
    {"epoch": 0, "step": 75, "mean_loss": 4.3, "mid_epoch": True,
     "median_error_km": 330.0, "external_median_error_km": 1180.0, "external_sign_z": 1.7},
    {"epoch": 0, "step": 107, "mean_loss": 4.2,
     "median_error_km": 324.0, "external_median_error_km": 1210.0, "external_sign_z": 1.5},
]


def test_mid_epoch_point_can_win_selection():
    """The commons01 optimum was inside epoch 0; step probing must be able to catch it."""
    selector = CheckpointSelector(name="external", metric_key="external_median_error_km")
    _run(selector, DENSE)
    assert selector.step == 50, "a mid-epoch step should be selectable, not just epoch ends"
    assert selector.epoch == 0


def test_dense_and_epoch_points_render_together():
    table = format_dual_trajectory(DENSE)
    for step in (25, 50, 75, 107):
        assert str(step) in table


def test_dense_probing_finds_an_optimum_epoch_granularity_would_miss():
    """With only the epoch-end point, step 50's better value is invisible."""
    epoch_only = CheckpointSelector(name="external", metric_key="external_median_error_km")
    _run(epoch_only, [p for p in DENSE if not p.get("mid_epoch")])
    dense = CheckpointSelector(name="external", metric_key="external_median_error_km")
    _run(dense, DENSE)
    assert dense.value < epoch_only.value


# --- early-stopping patience under dense probing -------------------------
#
# The osv5m01 bug: patience was checked against `stale_evals` (probe events).
# Dense probing every 10 steps put 33 probes in epoch 0, 18 of them
# non-improving, so a patience of 3 was exhausted before epoch 1 ever started.


def _dense_epoch(selector: CheckpointSelector, values: list[float], epoch: int, base_step: int):
    """Feed one epoch's worth of dense probes, then close the epoch."""
    for i, value in enumerate(values):
        selector.offer(
            {"external_median_error_km": value},
            epoch=epoch,
            step=base_step + i * 10,
            save=lambda name: name,
        )
    return selector.end_epoch()


def test_dense_probing_does_not_exhaust_patience_within_one_epoch():
    """The osv5m01 bug, reproduced: 18 non-improving probes inside epoch 0."""
    selector = CheckpointSelector(
        name="external", metric_key="external_median_error_km", drives_early_stop=True
    )
    # Improves early, then 18 probes that do not beat the best. This is the
    # real shape of osv5m01's epoch 0.
    values = [1300.0, 1200.0, 1100.0, 1043.0] + [1050.0 + i for i in range(18)]
    stale_epochs = _dense_epoch(selector, values, epoch=0, base_step=10)

    assert selector.stale_evals == 18, "probe-level counter still tracks probes"
    assert stale_epochs == 0, (
        "the epoch DID improve, so it must not count as a stale epoch -- "
        "this is the bug that truncated osv5m01 after epoch 0"
    )
    assert selector.stale_epochs < 3, "patience of 3 must not be spent inside one epoch"


def test_patience_now_counts_epochs_not_probes():
    selector = CheckpointSelector(
        name="external", metric_key="external_median_error_km", drives_early_stop=True
    )
    # Epoch 0 improves.
    assert _dense_epoch(selector, [1300.0, 1100.0], epoch=0, base_step=10) == 0
    # Three consecutive epochs of dense probes that never beat 1100.
    assert _dense_epoch(selector, [1200.0] * 20, epoch=1, base_step=100) == 1
    assert _dense_epoch(selector, [1150.0] * 20, epoch=2, base_step=200) == 2
    assert _dense_epoch(selector, [1180.0] * 20, epoch=3, base_step=300) == 3
    assert selector.stale_epochs == 3, "three stale EPOCHS should finally reach patience"


def test_improvement_late_in_an_epoch_resets_the_epoch_clock():
    selector = CheckpointSelector(
        name="external", metric_key="external_median_error_km", drives_early_stop=True
    )
    _dense_epoch(selector, [1300.0], epoch=0, base_step=10)
    _dense_epoch(selector, [1400.0] * 10, epoch=1, base_step=100)
    assert selector.stale_epochs == 1
    # A single improvement on the last probe of the epoch clears it.
    _dense_epoch(selector, [1400.0] * 9 + [1200.0], epoch=2, base_step=200)
    assert selector.stale_epochs == 0


def test_unmeasured_epoch_spends_no_patience():
    """A coarser external cadence must not be read as non-improvement."""
    selector = CheckpointSelector(
        name="external", metric_key="external_median_error_km", drives_early_stop=True
    )
    _dense_epoch(selector, [1300.0], epoch=0, base_step=10)
    # An epoch where only the in-distribution metric was measured.
    selector.offer({"median_error_km": 300.0}, epoch=1, step=100, save=lambda n: n)
    assert selector.end_epoch() == 0, "no external measurement means no evidence either way"


def test_coarse_per_epoch_path_is_unchanged():
    """batch01/breadth/commons01 behaviour: one eval per epoch, patience in epochs.

    With a single evaluation per epoch the new epoch clock must agree exactly
    with the old probe clock, which is what those earlier runs relied on.
    """
    selector = CheckpointSelector(
        name="indist", metric_key="median_error_km", drives_early_stop=True
    )
    values = [324.0, 320.0, 321.0, 322.0, 323.0]  # improves twice, then stalls
    for epoch, value in enumerate(values):
        selector.offer({"median_error_km": value}, epoch=epoch, step=epoch * 100,
                       save=lambda n: n)
        selector.end_epoch()
    assert selector.epoch == 1, "best is still epoch 1"
    assert selector.stale_epochs == 3, "three stale epochs, exactly as stale_evals used to be"
    assert selector.stale_epochs == selector.stale_evals, (
        "with one eval per epoch the two clocks must agree; that equivalence is "
        "what keeps the earlier coarse runs reproducible"
    )
