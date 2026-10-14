# r-analysis — statistical layer for View Finder's training pipeline

A standalone R project that sits **next to** `finetune/` and reads its
output — it never imports, calls, or modifies any Python code. Its only
input is the `summary.json` file every training run already writes to
`finetune/work/<run_name>/summary.json`; its only output is files under
`r-analysis/output/`.

## Why this exists

The Python pipeline (`finetune/`) picks the best checkpoint per run and
reports point-estimate metrics (median error, accuracy@threshold) for
that run. It does not:

- fit a proper dose-response curve across the L2-SP regularization sweep
  (0.001 → 30), with confidence intervals on the fitted parameters
- formally separate "this shard is genuinely harder/easier" from
  "this run just got a noisy external-benchmark draw" when runs disagree
- put a confidence interval on the current best model's benchmark
  accuracy, rather than a bare point estimate on 1000 external images

This project is that statistical rigor layer, run on demand, entirely
decoupled from training so it can never affect it.

## What it computes

1. **Dose-response curve** (`R/dose_response.R`) — fits a 4-parameter
   log-logistic dose-response model (via the `drc` package, falling back
   to `nls()` or a log-linear `lm()` if too few dose points are on record)
   relating L2-SP `reg_strength` to external-benchmark median error.
   Reports fitted parameters with confidence intervals and plots the
   curve + confidence band over the actual data (log-scale x-axis).

2. **Shard-to-shard variance decomposition** (`R/variance_analysis.R`) —
   fits an `lme4` random-intercept mixed model over the tail (converged
   portion) of each plain shard run's `external_trend` history, splitting
   the spread in external error into a between-shard component (real
   difference) and a residual component (noise), reported as plain-English
   percentages.

3. **Bootstrapped confidence intervals** (in `analyze_experiments.R`) —
   for the current best model (lowest external median error on record),
   bootstraps the accuracy-at-threshold metrics (25/200/750/2500 km) by
   resampling the reconstructed hit/miss vector from the reported n=1000
   external sample, 10,000 replicates, percentile CIs.

## Install

```bash
# From r-analysis/, in R or Rscript:
Rscript -e 'install.packages(c("jsonlite","dplyr","purrr","tibble","stringr","tidyr","ggplot2","scales","drc","lme4"), repos="https://cloud.r-project.org")'
```

`analyze_experiments.R` also checks for these packages itself and installs
anything missing on first run, so a bare `Rscript analyze_experiments.R`
works on a clean R install too.

We use a plain `install.packages()` flow rather than `renv` here: this is
a single-developer, read-only analysis tool re-run against a fast-moving
`finetune/work/` directory, not a shared deployed service, so a lockfile's
reproducibility guarantee wasn't worth the extra setup friction. If this
project grows collaborators or a CI job, `renv::init()` here would be the
natural next step — it needs no changes to the analysis code.

## Running it

```bash
cd r-analysis
Rscript analyze_experiments.R
```

Every run:
- re-scans `finetune/work/*/summary.json` from scratch — a new experiment
  directory (new regularization strength, new shard, whatever) is picked
  up automatically, no code changes needed
- writes a dated report to `output/report_<timestamp>.md`
- writes dated PNGs to `output/` (dose-response curve, shard variance,
  bootstrap CIs)
- prints a short console summary of the headline findings

**Re-run this after every new `finetune/work/<run>/summary.json` lands** —
a completed training run, a new regularization-strength test, a new shard.
Nothing needs to be reconfigured; the loader in `R/load_data.R` just reads
whatever's on disk.

## Layout

```
r-analysis/
  analyze_experiments.R   # main entry point — run this
  R/
    load_data.R            # parses finetune/work/*/summary.json into a tidy table
    dose_response.R        # Analysis 1: L2-SP dose-response curve fitting
    variance_analysis.R    # Analysis 2: lme4 shard-to-shard variance decomposition
    plots.R                # ggplot2 visualizations
  output/                  # generated reports + plots land here (gitignored contents)
  README.md
```

## Data source and important caveats

- **Dose-response family filtering**: `reg_strength` is only comparable
  across runs that share the same shard set and starting checkpoint (a
  regularization strength of 1.0 means something different starting from
  a different anchor). The script automatically picks the shard/anchor
  family with the most distinct dose levels on record as the "primary"
  sweep to fit, and plots any other-family points as supplementary
  (triangle markers), excluded from the fit.
- **Variance decomposition's "noise"**: `finetune/` doesn't log repeated
  independent evaluations at a fixed checkpoint, so this analysis uses the
  tail of each run's `external_trend` (evaluations every few steps, once
  training has plateaued) as a proxy for repeated measurement. A step
  fixed effect is included to soak up any residual drift within that tail
  window, but this is still evaluation-during-training jitter, not a
  from-scratch repeated eval — read the residual variance as an upper
  bound on true measurement noise, not a clean estimate of it.
- **Bootstrap CIs**: `summary.json` stores accuracy as a proportion over
  n=1000 external images, not the per-image hit/miss list. The bootstrap
  reconstructs a hit/miss vector consistent with the reported proportion
  and resamples from that — mathematically equivalent to bootstrapping
  the original Bernoulli sample, but it means the CI reflects sampling
  uncertainty only, not any rounding in how the original proportion was
  reported.
