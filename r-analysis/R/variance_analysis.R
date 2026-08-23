#' Analysis 2: shard-to-shard variance decomposition.
#'
#' Every un-regularized shard fine-tune (osv5m01..osv5m07 and any future
#' single-shard run in the same family) logs an `external_trend`: a series
#' of external-benchmark evaluations taken every few training steps. Deep
#' in each run's training (once external error has stopped trending down),
#' those repeated evaluations are the closest thing on disk to repeated
#' measurements of "how good is this shard's fine-tune, really" -- so we
#' use the tail of each run's trend as within-shard replicate measurements
#' and fit a random-intercept mixed model with lme4 to split their spread
#' into a shard-to-shard (between-run) component and a noise (residual)
#' component.
#'
#' Caveat this analysis is upfront about: trend points a few steps apart
#' are not independent draws in the textbook sense (the same external
#' sample is scored repeatedly as the model keeps training), so the
#' "residual" component here is best read as step-to-step measurement
#' jitter conditional on the training trajectory, not pure sampling noise.
#' We partial out any residual drift within the tail window with a step
#' fixed effect so the random-intercept variance isn't inflated by
#' leftover trend.

suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tibble)
})

#' Identify "plain shard" runs: single, non-combined shard id, no L2-SP
#' regularization applied. This is what makes osv5m01..osv5m07 (and any
#' future osv5m0N) comparable as repeated draws of "fine-tune on one fresh
#' shard starting from the shared pipeline."
build_variance_data <- function(runs, tail_fraction = 0.4, min_tail_rows = 5) {
  shard_runs <- runs |>
    dplyr::filter(
      is.na(reg_strength),
      !is.na(shards),
      !stringr::str_detect(shards, ","),
      purrr::map_lgl(external_trend, ~ is.data.frame(.x) && nrow(.x) >= min_tail_rows)
    )

  if (nrow(shard_runs) < 2) {
    return(tibble::tibble())
  }

  shard_runs |>
    dplyr::mutate(
      tail = purrr::map(external_trend, function(tr) {
        tr <- tr[order(tr$step), , drop = FALSE]
        n <- nrow(tr)
        keep_n <- max(min_tail_rows, ceiling(n * tail_fraction))
        tail(tr, keep_n)
      })
    ) |>
    dplyr::select(run_name, shards, tail) |>
    tidyr::unnest(tail) |>
    dplyr::mutate(
      step_scaled = as.numeric(scale(step))
    ) |>
    dplyr::filter(!is.na(median_error_km))
}

#' Fit the random-intercept mixed model and return variance components in
#' both raw and plain-language form.
fit_variance_model <- function(var_data) {
  if (nrow(var_data) == 0 || dplyr::n_distinct(var_data$run_name) < 2) {
    return(list(fit = NULL, summary = NULL))
  }

  fit <- tryCatch(
    lme4::lmer(median_error_km ~ step_scaled + (1 | run_name), data = var_data, REML = TRUE),
    error = function(e) NULL
  )
  used_step_trend <- TRUE
  if (is.null(fit) || lme4::isSingular(fit)) {
    fallback <- tryCatch(
      lme4::lmer(median_error_km ~ (1 | run_name), data = var_data, REML = TRUE),
      error = function(e) NULL
    )
    if (!is.null(fallback)) {
      fit <- fallback
      used_step_trend <- FALSE
    }
  }
  if (is.null(fit)) return(list(fit = NULL, summary = NULL))

  vc <- as.data.frame(lme4::VarCorr(fit))
  between_var <- vc$vcov[vc$grp == "run_name"]
  resid_var <- vc$vcov[vc$grp == "Residual"]
  total_var <- between_var + resid_var
  icc <- between_var / total_var

  n_runs <- dplyr::n_distinct(var_data$run_name)
  n_obs <- nrow(var_data)

  list(
    fit = fit,
    used_step_trend = used_step_trend,
    n_runs = n_runs,
    n_obs = n_obs,
    between_shard_sd_km = sqrt(between_var),
    residual_sd_km = sqrt(resid_var),
    pct_shard_to_shard = 100 * icc,
    pct_noise = 100 * (1 - icc),
    plain_summary = sprintf(
      paste0(
        "Across %d shard fine-tunes (%d tail-window evaluations), %.0f%% of the ",
        "variation in external median error is real shard-to-shard difference ",
        "(between-run SD = %.1f km), and %.0f%% is step-to-step noise around each ",
        "shard's own plateau (residual SD = %.1f km)."
      ),
      n_runs, n_obs, 100 * icc, sqrt(between_var), 100 * (1 - icc), sqrt(resid_var)
    )
  )
}
