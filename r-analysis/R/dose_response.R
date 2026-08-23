#' Analysis 1: L2-SP regularization-strength dose-response curve.
#'
#' Relates L2-SP `reg_strength` to external-benchmark median error (km).
#' Fits a four-parameter log-logistic dose-response model (the standard
#' pharmacology-style curve: two asymptotes, an EC50-equivalent inflection
#' dose, and a slope) via the `drc` package, with an `nls()` fallback if
#' `drc` can't be used (e.g. too few distinct doses to identify 4 params).

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

#' Build the dose-response dataset: every run with a recorded reg_strength.
#' Runs are tagged with their (shards, anchor_run) "family" so the caller
#' can restrict fitting to a single controlled sweep -- reg_strength only
#' means the same thing when the base data and starting checkpoint match.
build_dose_response_data <- function(runs) {
  runs |>
    dplyr::filter(!is.na(reg_strength), !is.na(ext_sel_median_km)) |>
    dplyr::mutate(family = paste(shards, anchor_run, sep = " | ")) |>
    dplyr::select(
      run_name, reg_strength, shards, anchor_run, family,
      external_error_km = ext_sel_median_km,
      external_n = ext_sel_n
    ) |>
    dplyr::arrange(reg_strength)
}

#' Pick the dose-response family (shard set + anchor checkpoint) with the
#' most distinct dose levels -- that's the controlled sweep worth curve
#' fitting. Other families are returned separately as supplementary points.
select_primary_family <- function(dr_data) {
  counts <- dr_data |>
    dplyr::group_by(family) |>
    dplyr::summarise(n_doses = dplyr::n_distinct(reg_strength), .groups = "drop") |>
    dplyr::arrange(dplyr::desc(n_doses))
  best_family <- counts$family[[1]]
  list(
    primary = dplyr::filter(dr_data, family == best_family),
    supplementary = dplyr::filter(dr_data, family != best_family),
    family_label = best_family
  )
}

#' Case-resampling bootstrap for the LL.4 dose-response fit. Used as a
#' fallback when the model's asymptotic (Hessian-based) standard errors are
#' NaN -- which happens here because several high-dose points land on
#' (near-)identical external error, making the curve's asymptote region
#' locally flat and the delta-method covariance matrix singular. The
#' bootstrap doesn't need that Hessian to be well-behaved, only that enough
#' resamples converge to characterize the sampling distribution.
bootstrap_ll4 <- function(data, grid, param_names, n_boot = 2000, seed = 1337) {
  set.seed(seed)
  n <- nrow(data)
  coef_draws <- vector("list", n_boot)
  pred_draws <- vector("list", n_boot)
  n_success <- 0
  for (i in seq_len(n_boot)) {
    boot_rows <- data[sample.int(n, n, replace = TRUE), ]
    if (dplyr::n_distinct(boot_rows$reg_strength) < 4) next
    boot_fit <- tryCatch(
      drc::drm(external_error_km ~ reg_strength, data = boot_rows, fct = drc::LL.4(names = param_names)),
      error = function(e) NULL, warning = function(w) NULL
    )
    if (is.null(boot_fit)) next
    co <- tryCatch(coef(boot_fit), error = function(e) NULL)
    if (is.null(co) || anyNA(co)) next
    pred <- tryCatch(as.numeric(predict(boot_fit, newdata = grid)), error = function(e) NULL)
    if (is.null(pred) || anyNA(pred)) next
    n_success <- n_success + 1
    coef_draws[[n_success]] <- co
    pred_draws[[n_success]] <- pred
  }
  coef_mat <- do.call(rbind, coef_draws[seq_len(n_success)])
  pred_mat <- do.call(rbind, pred_draws[seq_len(n_success)])
  list(
    n_boot = n_boot,
    n_success = n_success,
    coef_ci = apply(coef_mat, 2, quantile, probs = c(0.025, 0.975)),
    coef_sd = apply(coef_mat, 2, sd),
    pred_ci = if (!is.null(pred_mat)) apply(pred_mat, 2, quantile, probs = c(0.025, 0.975)) else NULL
  )
}

#' Fit a 4-parameter log-logistic dose-response curve:
#'   error(dose) = c + (d - c) / (1 + exp(b * (log(dose) - log(e))))
#' d = upper asymptote (error at vanishing regularization)
#' c = lower asymptote (error at saturating regularization)
#' e = inflection dose (EC50-equivalent)
#' b = slope at the inflection
#'
#' Returns a list with the fitted model, a tidy coefficient/CI table, a
#' method flag ("drc" or "nls_LL4" or "nls_linear_log"), and (if enough
#' data) a data.frame of predictions across the dose range with a 95%
#' confidence band suitable for plotting.
fit_dose_response_curve <- function(dr_primary) {
  x <- dr_primary$reg_strength
  y <- dr_primary$external_error_km
  n_doses <- dplyr::n_distinct(x)

  grid <- data.frame(reg_strength = exp(seq(log(min(x)), log(max(x)), length.out = 200)))

  # --- Preferred path: drc::drm with a 4-parameter log-logistic model ---
  param_names <- c("slope", "lower_asymptote", "upper_asymptote", "ec50")
  if (requireNamespace("drc", quietly = TRUE) && n_doses >= 4) {
    fit <- tryCatch(
      drc::drm(external_error_km ~ reg_strength, data = dr_primary, fct = drc::LL.4(names = param_names)),
      error = function(e) NULL
    )
    if (!is.null(fit)) {
      coefs <- summary(fit)$coefficients
      ci <- tryCatch(confint(fit), error = function(e) matrix(NA, nrow(coefs), 2))
      coef_table <- tibble::tibble(
        parameter = rownames(coefs),
        estimate = coefs[, "Estimate"],
        std_error = coefs[, "Std. Error"],
        p_value = coefs[, "p-value"],
        ci_low = ci[, 1],
        ci_high = ci[, 2]
      )
      pred <- tryCatch(
        as.data.frame(predict(fit, newdata = grid, interval = "confidence")),
        error = function(e) NULL
      )
      pred_df <- NULL
      if (!is.null(pred)) {
        pred_df <- tibble::tibble(
          reg_strength = grid$reg_strength,
          fit = pred[, 1],
          ci_low = pred[, 2],
          ci_high = pred[, 3]
        )
      }

      ci_note <- "asymptotic (delta-method) confidence intervals from drc::drm"
      boot_info <- NULL
      if (any(is.nan(coef_table$std_error)) || any(is.na(coef_table$ci_low))) {
        boot_info <- tryCatch(bootstrap_ll4(dr_primary, grid, param_names, n_boot = 800), error = function(e) NULL)
        names_ok <- !is.null(boot_info) && !is.null(boot_info$coef_ci) &&
          all(coef_table$parameter %in% colnames(boot_info$coef_ci))
        if (!is.null(boot_info) && boot_info$n_success >= 200 && names_ok) {
          coef_table$std_error <- as.numeric(boot_info$coef_sd[coef_table$parameter])
          coef_table$ci_low <- as.numeric(boot_info$coef_ci["2.5%", coef_table$parameter])
          coef_table$ci_high <- as.numeric(boot_info$coef_ci["97.5%", coef_table$parameter])
          coef_table$p_value <- NA_real_
          ci_note <- sprintf(
            "asymptotic SEs were singular (high-dose points plateau at an identical value); used a %d-replicate case-resampling bootstrap instead (%d/%d resamples converged)",
            boot_info$n_boot, boot_info$n_success, boot_info$n_boot
          )
          if (!is.null(boot_info$pred_ci) && !is.null(pred_df) && ncol(boot_info$pred_ci) == nrow(pred_df)) {
            pred_df$ci_low <- as.numeric(boot_info$pred_ci["2.5%", ])
            pred_df$ci_high <- as.numeric(boot_info$pred_ci["97.5%", ])
          }
        } else {
          n_succ <- if (is.null(boot_info)) 0 else boot_info$n_success
          n_tot <- if (is.null(boot_info)) 800 else boot_info$n_boot
          ci_note <- sprintf(
            "asymptotic SEs were singular and the case-resampling bootstrap could not produce reliable CIs (%d/%d resamples usable) -- point estimates only",
            n_succ, n_tot
          )
        }
      }

      return(list(
        method = "drc_LL4",
        model = fit,
        coef_table = coef_table,
        predictions = pred_df,
        n_doses = n_doses,
        ci_note = ci_note
      ))
    }
  }

  # --- Fallback: hand-rolled nls() 4-parameter log-logistic ---
  fit <- tryCatch({
    start <- list(
      c = min(y), d = max(y),
      e = stats::median(x), b = 1
    )
    nls(
      y ~ c + (d - c) / (1 + exp(b * (log(x) - log(e)))),
      data = data.frame(x = x, y = y),
      start = start,
      control = nls.control(maxiter = 200, warnOnly = TRUE)
    )
  }, error = function(e) NULL)

  if (!is.null(fit)) {
    co <- summary(fit)$coefficients
    ci <- tryCatch(confint(fit), error = function(e) {
      se <- co[, "Std. Error"]
      cbind(co[, "Estimate"] - 1.96 * se, co[, "Estimate"] + 1.96 * se)
    })
    coef_table <- tibble::tibble(
      parameter = rownames(co),
      estimate = co[, "Estimate"],
      std_error = co[, "Std. Error"],
      p_value = co[, 4],
      ci_low = ci[, 1],
      ci_high = ci[, 2]
    )
    fit_vals <- tryCatch(predict(fit, newdata = data.frame(x = grid$reg_strength)), error = function(e) NULL)
    pred_df <- if (!is.null(fit_vals)) {
      tibble::tibble(reg_strength = grid$reg_strength, fit = fit_vals, ci_low = NA_real_, ci_high = NA_real_)
    } else NULL
    return(list(
      method = "nls_LL4",
      model = fit,
      coef_table = coef_table,
      predictions = pred_df,
      n_doses = n_doses,
      ci_note = "asymptotic (Wald/profile) confidence intervals from nls()"
    ))
  }

  # --- Last resort: simple linear-in-log-dose fit, at least gives a trend ---
  fit <- lm(y ~ log(x), data = data.frame(x = x, y = y))
  co <- summary(fit)$coefficients
  ci <- confint(fit)
  coef_table <- tibble::tibble(
    parameter = rownames(co),
    estimate = co[, "Estimate"],
    std_error = co[, "Std. Error"],
    p_value = co[, 4],
    ci_low = ci[, 1],
    ci_high = ci[, 2]
  )
  pred <- predict(fit, newdata = data.frame(x = grid$reg_strength), interval = "confidence")
  pred_df <- tibble::tibble(
    reg_strength = grid$reg_strength,
    fit = pred[, "fit"], ci_low = pred[, "lwr"], ci_high = pred[, "upr"]
  )
  list(
    method = "lm_log_linear", model = fit, coef_table = coef_table, predictions = pred_df, n_doses = n_doses,
    ci_note = "asymptotic confidence intervals from lm() -- the log-logistic fit did not converge, this is a simpler log-linear trend line"
  )
}
