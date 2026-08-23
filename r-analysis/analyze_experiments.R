#!/usr/bin/env Rscript
#' analyze_experiments.R
#'
#' Re-runnable statistical analysis layer for View Finder's location-encoder
#' fine-tuning experiments. Reads finetune/work/*/summary.json (written by
#' the existing Python training pipeline -- untouched by this script) and
#' produces:
#'
#'   1. A dose-response fit of L2-SP regularization strength vs. external
#'      benchmark error.
#'   2. A shard-to-shard variance decomposition (lme4 mixed model) that
#'      splits observed run-to-run spread into real shard differences vs.
#'      noise.
#'   3. Bootstrapped confidence intervals for the current best model's
#'      benchmark accuracy at each distance threshold.
#'
#' Run it after any new finetune/work/<run>/summary.json appears -- no code
#' changes needed, it just re-scans the directory.
#'
#'   Rscript analyze_experiments.R

set.seed(1337)

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) NULL
)
if (is.null(script_dir) || script_dir == "") script_dir <- getwd()
setwd(script_dir)

required_pkgs <- c(
  "jsonlite", "dplyr", "purrr", "tibble", "stringr", "tidyr",
  "ggplot2", "scales", "drc", "lme4"
)
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(tibble)
  library(stringr)
  library(ggplot2)
})

source("R/load_data.R")
source("R/dose_response.R")
source("R/variance_analysis.R")
source("R/plots.R")

dir.create("output", showWarnings = FALSE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

cat("== View Finder R statistical analysis layer ==\n")
cat("Loading runs from finetune/work/ ...\n")
runs <- load_all_runs()
cat(sprintf("Loaded %d runs.\n\n", nrow(runs)))

# ---------------------------------------------------------------------------
# Analysis 1: dose-response curve
# ---------------------------------------------------------------------------
cat("-- Analysis 1: L2-SP dose-response --\n")
dr_data <- build_dose_response_data(runs)
dr_split <- select_primary_family(dr_data)
dr_fit <- fit_dose_response_curve(dr_split$primary)

cat(sprintf(
  "  %d dose points in primary family (%s); %d supplementary points from other families.\n",
  nrow(dr_split$primary), dr_split$family_label, nrow(dr_split$supplementary)
))
cat(sprintf("  Fit method: %s\n", dr_fit$method))
cat(sprintf("  CI method: %s\n", dr_fit$ci_note))
print(dr_fit$coef_table, n = Inf)

dose_response_plot <- plot_dose_response(dr_split$primary, dr_split$supplementary, dr_fit)
dose_response_plot_path <- file.path("output", sprintf("dose_response_%s.png", timestamp))
ggsave(dose_response_plot_path, dose_response_plot, width = 8, height = 5.5, dpi = 150)
cat(sprintf("  Saved plot: %s\n\n", dose_response_plot_path))

# ---------------------------------------------------------------------------
# Analysis 2: shard-to-shard variance decomposition
# ---------------------------------------------------------------------------
cat("-- Analysis 2: shard-to-shard variance decomposition --\n")
var_data <- build_variance_data(runs)
var_result <- fit_variance_model(var_data)

variance_plot_path <- NULL
if (!is.null(var_result$fit)) {
  cat(" ", var_result$plain_summary, "\n")
  print(summary(var_result$fit))
  variance_plot <- plot_variance_by_shard(var_data)
  variance_plot_path <- file.path("output", sprintf("shard_variance_%s.png", timestamp))
  ggsave(variance_plot_path, variance_plot, width = 8, height = 5.5, dpi = 150)
  cat(sprintf("  Saved plot: %s\n\n", variance_plot_path))
} else {
  cat("  Not enough comparable shard runs with usable external_trend history to fit a mixed model.\n\n")
}

# ---------------------------------------------------------------------------
# Analysis 3: bootstrapped CIs for the current best model
# ---------------------------------------------------------------------------
cat("-- Analysis 3: bootstrapped CIs for the current best model --\n")

best_run <- runs |>
  filter(!is.na(ext_sel_median_km)) |>
  arrange(ext_sel_median_km) |>
  slice(1)

cat(sprintf(
  "  Current best model: %s (external median error = %.1f km, n = %d)\n",
  best_run$run_name, best_run$ext_sel_median_km, best_run$ext_sel_n
))

bootstrap_accuracy <- function(p, n, n_boot = 10000, seed = 1337) {
  set.seed(seed)
  hits <- round(p * n)
  sample_vec <- c(rep(1, hits), rep(0, n - hits))
  boot_props <- replicate(n_boot, mean(sample(sample_vec, n, replace = TRUE)))
  tibble::tibble(
    estimate = p,
    ci_low = as.numeric(quantile(boot_props, 0.025)),
    ci_high = as.numeric(quantile(boot_props, 0.975)),
    n = n,
    n_boot = n_boot
  )
}

thresholds <- tibble::tibble(
  threshold = c("25km", "200km", "750km", "2500km"),
  p = c(best_run$ext_sel_acc_25km, best_run$ext_sel_acc_200km, best_run$ext_sel_acc_750km, best_run$ext_sel_acc_2500km)
)
boot_summary <- thresholds |>
  mutate(boot = purrr::map(p, bootstrap_accuracy, n = best_run$ext_sel_n)) |>
  select(-p) |>
  tidyr::unnest(boot)

print(boot_summary, n = Inf)

bootstrap_plot_path <- file.path("output", sprintf("bootstrap_ci_%s.png", timestamp))
ggsave(bootstrap_plot_path, plot_bootstrap_ci(boot_summary), width = 7, height = 5, dpi = 150)
cat(sprintf("  Saved plot: %s\n\n", bootstrap_plot_path))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report_path <- file.path("output", sprintf("report_%s.md", timestamp))

fmt_num <- function(x, digits = 3) formatC(x, digits = digits, format = "g")

coef_md_table <- function(tbl) {
  header <- "| parameter | estimate | std_error | p_value | ci_low | ci_high |"
  sep <- "|---|---|---|---|---|---|"
  rows <- apply(tbl, 1, function(r) {
    sprintf(
      "| %s | %s | %s | %s | %s | %s |",
      r[["parameter"]], fmt_num(as.numeric(r[["estimate"]])), fmt_num(as.numeric(r[["std_error"]])),
      fmt_num(as.numeric(r[["p_value"]])), fmt_num(as.numeric(r[["ci_low"]])), fmt_num(as.numeric(r[["ci_high"]]))
    )
  })
  paste(c(header, sep, rows), collapse = "\n")
}

boot_md_table <- function(tbl) {
  header <- "| threshold | estimate | ci_low | ci_high |"
  sep <- "|---|---|---|---|"
  rows <- apply(tbl, 1, function(r) {
    sprintf(
      "| %s | %.1f%% | %.1f%% | %.1f%% |",
      r[["threshold"]], 100 * as.numeric(r[["estimate"]]),
      100 * as.numeric(r[["ci_low"]]), 100 * as.numeric(r[["ci_high"]])
    )
  })
  paste(c(header, sep, rows), collapse = "\n")
}

variance_section <- if (!is.null(var_result$fit)) {
  sprintf(
    paste0(
      "%s\n\n",
      "- Shard runs used: %d (%d tail-window evaluations)\n",
      "- Between-shard SD: %.1f km\n",
      "- Residual (noise) SD: %.1f km\n",
      "- Share of variance that is real shard-to-shard difference: %.0f%%\n",
      "- Share of variance that is noise: %.0f%%\n\n",
      "![shard variance](%s)\n"
    ),
    var_result$plain_summary, var_result$n_runs, var_result$n_obs,
    var_result$between_shard_sd_km, var_result$residual_sd_km,
    var_result$pct_shard_to_shard, var_result$pct_noise,
    basename(variance_plot_path)
  )
} else {
  "Not enough comparable shard runs with usable `external_trend` history to fit a mixed model yet."
}

report <- sprintf(
'# View Finder training experiments -- statistical analysis report

Generated: %s
Runs analyzed: %d (from `finetune/work/*/summary.json`)

This report is produced by `r-analysis/analyze_experiments.R`, a read-only
statistical layer over the Python training pipeline. It does not affect
training in any way -- rerun it any time a new experiment finishes.

## 1. L2-SP dose-response

Primary sweep family: `%s` (%d dose points). %d supplementary points from
other shard/anchor combinations are plotted but excluded from the curve fit,
since `reg_strength` is only comparable within a fixed base dataset and
starting checkpoint.

Fit method: **%s**
CI method: %s

%s

![dose response](%s)

## 2. Shard-to-shard variance decomposition

%s

## 3. Bootstrapped confidence intervals -- current best model

Current best model (by external median error): **%s**
External median error: %.1f km (n = %d)

%s

![bootstrap CIs](%s)

## Appendix: all runs on record

%s
',
  format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), nrow(runs),
  dr_split$family_label, nrow(dr_split$primary), nrow(dr_split$supplementary),
  dr_fit$method, dr_fit$ci_note, coef_md_table(dr_fit$coef_table), basename(dose_response_plot_path),
  variance_section,
  best_run$run_name, best_run$ext_sel_median_km, best_run$ext_sel_n,
  boot_md_table(boot_summary), basename(bootstrap_plot_path),
  paste(capture.output(print(
    runs |> select(run_name, dataset, shards, reg_strength, ext_sel_median_km, ext_sel_n) |> as.data.frame(),
    row.names = FALSE
  )), collapse = "\n")
)

writeLines(report, report_path)
cat(sprintf("\nReport written to: %s\n", report_path))

cat("\n== Summary ==\n")
cat(sprintf(
  "Dose-response: %d doses fit via %s. See coefficient table above / in report.\n",
  nrow(dr_split$primary), dr_fit$method
))
if (!is.null(var_result$fit)) {
  cat(" ", var_result$plain_summary, "\n")
}
cat(sprintf(
  "Best model %s: acc@25km %.1f%% [%.1f, %.1f], acc@200km %.1f%% [%.1f, %.1f], acc@750km %.1f%% [%.1f, %.1f], acc@2500km %.1f%% [%.1f, %.1f]\n",
  best_run$run_name,
  100 * boot_summary$estimate[1], 100 * boot_summary$ci_low[1], 100 * boot_summary$ci_high[1],
  100 * boot_summary$estimate[2], 100 * boot_summary$ci_low[2], 100 * boot_summary$ci_high[2],
  100 * boot_summary$estimate[3], 100 * boot_summary$ci_low[3], 100 * boot_summary$ci_high[3],
  100 * boot_summary$estimate[4], 100 * boot_summary$ci_low[4], 100 * boot_summary$ci_high[4]
))
