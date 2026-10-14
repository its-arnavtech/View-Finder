"""Selecting a checkpoint on more than one metric at once.

Why this exists
---------------
The training loop used to track a single scalar and save one
``location_encoder_best.pth``. That silently assumed in-distribution held-out
error is a proxy for the thing the project actually wants, which is
generalization to arbitrary images. The ``commons01`` run showed the two are
not merely imperfectly correlated -- over the run they moved in **opposite**
directions:

===== =================== ==================
epoch held-out median km  external median km
===== =================== ==================
0     324.0               1164.6
2     320.7               1199.1
4     **314.3** (best)    1453.4
6     326.9               1478.7
===== =================== ==================

Held-out error bottomed at epoch 4, so save-best chose epoch 4 -- which is
roughly the *worst* point of the run externally. Selecting on the
in-distribution metric actively selected against the goal.

So selection is now plural. Each :class:`CheckpointSelector` owns one metric,
one best-so-far, and one checkpoint file, and exactly one of them drives early
stopping. Both are always written and always reported, because the divergence
between them is itself the finding worth seeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CheckpointSelector:
    """Tracks the best value of one metric and owns one checkpoint file.

    Attributes:
        name: Short id, used in the filename and the report.
        metric_key: Key to read from the metrics dict an eval hook returns.
        drives_early_stop: Whether non-improvement here counts toward patience.
            Exactly one selector in a run should have this set.
        lower_is_better: All current metrics are errors, but the sign-test z is
            not, so this is explicit rather than assumed.
    """

    name: str
    metric_key: str
    drives_early_stop: bool = False
    lower_is_better: bool = True
    description: str = ""

    value: float = field(default=float("inf"), init=False)
    epoch: int | None = field(default=None, init=False)
    step: int | None = field(default=None, init=False)
    path: str | None = field(default=None, init=False)
    metrics: dict | None = field(default=None, init=False)

    # Two clocks, deliberately separate.
    #
    # `stale_evals` counts probe events since the last improvement. It is a
    # diagnostic only, and must NOT drive early stopping: under dense probing a
    # single epoch can contain dozens of probes, so patience expressed in epochs
    # would be spent within one epoch. That is exactly what truncated osv5m01 --
    # 33 probes in epoch 0, 18 of them non-improving, against a patience of 3,
    # so epochs 1-11 never ran.
    #
    # `stale_epochs` counts *epochs* that were measured and produced no
    # improvement. Early stopping uses this. Dense probing is about locating the
    # optimum precisely; early stopping is about deciding training has stopped
    # helping. Different questions on different timescales.
    stale_evals: int = field(default=0, init=False)
    stale_epochs: int = field(default=0, init=False)
    improved_this_epoch: bool = field(default=False, init=False)
    measured_this_epoch: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.lower_is_better:
            self.value = float("-inf")

    @property
    def filename(self) -> str:
        return f"location_encoder_best_{self.name}.pth"

    def _improves(self, candidate: float) -> bool:
        return candidate < self.value if self.lower_is_better else candidate > self.value

    def offer(
        self,
        metrics: dict,
        *,
        epoch: int,
        step: int,
        save: Callable[[str], str],
    ) -> bool:
        """Consider one evaluation. Returns True when it became the new best.

        A metrics dict that does not carry this selector's key is *skipped*, not
        counted as a non-improvement: the external probe may run on a coarser
        cadence than the held-out eval, and a missing measurement is not
        evidence of a worse one.
        """
        if self.metric_key not in metrics or metrics[self.metric_key] is None:
            return False

        self.measured_this_epoch = True
        candidate = float(metrics[self.metric_key])
        if self._improves(candidate):
            self.value = candidate
            self.epoch = epoch
            self.step = step
            self.metrics = dict(metrics)
            self.path = save(self.filename)
            self.stale_evals = 0
            self.improved_this_epoch = True
            return True

        self.stale_evals += 1
        return False

    def end_epoch(self) -> int:
        """Close an epoch and return epochs-without-improvement.

        An epoch counts as stale only if this selector was actually measured
        during it and never improved. An epoch where the metric was never
        measured (coarser probe cadence) spends no patience, for the same
        reason a missing measurement does not in :meth:`offer` -- an absent
        measurement is not evidence of a worse one.
        """
        if self.measured_this_epoch and not self.improved_this_epoch:
            self.stale_epochs += 1
        elif self.improved_this_epoch:
            self.stale_epochs = 0
        self.improved_this_epoch = False
        self.measured_this_epoch = False
        return self.stale_epochs

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_key": self.metric_key,
            "drives_early_stop": self.drives_early_stop,
            "best_value": None if self.value in (float("inf"), float("-inf")) else self.value,
            "best_epoch": self.epoch,
            "best_step": self.step,
            "path": self.path,
            "description": self.description,
        }


def legacy_selector(metric_key: str = "median_error_km") -> CheckpointSelector:
    """The single in-distribution selector, for callers not yet passing their own."""
    selector = CheckpointSelector(
        name="indist",
        metric_key=metric_key,
        drives_early_stop=True,
        description="in-distribution held-out error",
    )
    return selector


# --- reporting -----------------------------------------------------------

def format_dual_trajectory(
    trajectory: list[dict],
    *,
    indist_key: str = "median_error_km",
    external_key: str = "external_median_error_km",
    external_z_key: str = "external_sign_z",
    external_baseline: float | None = None,
    indist_baseline: float | None = None,
) -> str:
    """Render both metrics side by side across every probe point.

    Printed at the end of every run so a divergence like commons01's is visible
    immediately rather than discovered by accident months later. The ``agree``
    column is the point: it marks whether the two metrics moved the same way
    since the previous probe.
    """
    if not trajectory:
        return "(no evaluation points recorded)"

    header = (
        f"{'epoch':>6} {'step':>6} {'train_loss':>11} | "
        f"{'indist_km':>10} {'d_indist':>9} | "
        f"{'extern_km':>10} {'d_extern':>9} {'sign_z':>7} | {'agree':>6}"
    )
    lines = [header, "-" * len(header)]

    previous_indist: float | None = None
    previous_external: float | None = None
    best_indist = min(
        (p[indist_key] for p in trajectory if p.get(indist_key) is not None),
        default=None,
    )
    best_external = min(
        (p[external_key] for p in trajectory if p.get(external_key) is not None),
        default=None,
    )

    for point in trajectory:
        indist = point.get(indist_key)
        external = point.get(external_key)
        z = point.get(external_z_key)

        d_indist = "" if indist is None or indist_baseline is None else f"{indist - indist_baseline:+.1f}"
        d_external = "" if external is None or external_baseline is None else f"{external - external_baseline:+.1f}"

        # Did the two metrics move the same direction since the last probe?
        agree = ""
        if (
            indist is not None and external is not None
            and previous_indist is not None and previous_external is not None
        ):
            indist_better = indist < previous_indist
            external_better = external < previous_external
            agree = "yes" if indist_better == external_better else "NO"

        mark = ""
        if indist is not None and best_indist is not None and indist == best_indist:
            mark += "I"
        if external is not None and best_external is not None and external == best_external:
            mark += "E"

        lines.append(
            f"{point.get('epoch', ''):>6} {point.get('step', ''):>6} "
            f"{point.get('mean_loss', float('nan')):>11.5f} | "
            f"{'' if indist is None else f'{indist:10.2f}'} {d_indist:>9} | "
            f"{'' if external is None else f'{external:10.2f}'} {d_external:>9} "
            f"{'' if z is None else f'{z:7.2f}'} | {agree:>6} {mark}"
        )

        if indist is not None:
            previous_indist = indist
        if external is not None:
            previous_external = external

    lines.append("")
    lines.append("  I = best in-distribution point    E = best external point")
    lines.append("  agree = did both metrics move the same direction since the previous probe")
    lines.append("  'NO' rows are where selecting on in-distribution error picks against generalization.")
    return "\n".join(lines)


def divergence_summary(
    trajectory: list[dict],
    *,
    indist_key: str = "median_error_km",
    external_key: str = "external_median_error_km",
) -> str:
    """One-paragraph verdict on whether the two metrics agreed in this run."""
    points = [
        p for p in trajectory
        if p.get(indist_key) is not None and p.get(external_key) is not None
    ]
    if len(points) < 2:
        return "Not enough paired probe points to assess metric agreement."

    best_indist = min(points, key=lambda p: p[indist_key])
    best_external = min(points, key=lambda p: p[external_key])

    if best_indist["epoch"] == best_external["epoch"]:
        return (
            f"Both metrics agree: epoch {best_indist['epoch']} is best on "
            f"in-distribution ({best_indist[indist_key]:.2f} km) and external "
            f"({best_external[external_key]:.2f} km)."
        )

    external_at_indist_best = best_indist[external_key]
    penalty = external_at_indist_best - best_external[external_key]
    return (
        f"METRICS DIVERGE. Best in-distribution is epoch {best_indist['epoch']} "
        f"({best_indist[indist_key]:.2f} km), but best external is epoch "
        f"{best_external['epoch']} ({best_external[external_key]:.2f} km). "
        f"Selecting on in-distribution would ship a checkpoint that is "
        f"{penalty:+.2f} km worse externally than the best available "
        f"({external_at_indist_best:.2f} vs {best_external[external_key]:.2f} km)."
    )
