#' ggplot2 visualizations for the analysis report.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

theme_va <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )
}

#' Dose-response plot: log-scale x-axis, fitted curve + confidence band
#' overlaid on the observed (dose, external error) points.
plot_dose_response <- function(dr_primary, dr_supplementary, fit_result) {
  p <- ggplot()

  if (!is.null(fit_result$predictions)) {
    pred <- fit_result$predictions
    if (all(c("ci_low", "ci_high") %in% names(pred)) && !all(is.na(pred$ci_low))) {
      p <- p + geom_ribbon(
        data = pred, aes(x = reg_strength, ymin = ci_low, ymax = ci_high),
        fill = "#4C78A8", alpha = 0.15
      )
    }
    p <- p + geom_line(data = pred, aes(x = reg_strength, y = fit), color = "#4C78A8", linewidth = 1)
  }

  p <- p +
    geom_point(
      data = dr_primary, aes(x = reg_strength, y = external_error_km),
      size = 3, color = "#1B3A5C"
    )

  if (nrow(dr_supplementary) > 0) {
    p <- p + geom_point(
      data = dr_supplementary, aes(x = reg_strength, y = external_error_km),
      size = 2.5, shape = 17, color = "#B0413E"
    )
  }

  p +
    scale_x_log10() +
    labs(
      title = "L2-SP regularization strength vs. external benchmark error",
      subtitle = paste0(
        "Fitted with ", fit_result$method,
        " | circles = primary sweep, triangles = other shard/anchor family"
      ),
      x = "L2-SP regularization strength (log scale)",
      y = "External benchmark median error (km)"
    ) +
    theme_va()
}

#' Shard-to-shard variance plot: one point per tail-window observation,
#' colored/faceted by run, showing the spread the mixed model decomposes.
plot_variance_by_shard <- function(var_data) {
  ggplot(var_data, aes(x = run_name, y = median_error_km)) +
    geom_jitter(width = 0.15, alpha = 0.5, color = "#4C78A8") +
    stat_summary(fun = mean, geom = "point", size = 3, color = "#B0413E") +
    stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.25, color = "#B0413E") +
    labs(
      title = "Shard-to-shard spread in external error (tail-window evaluations)",
      subtitle = "Grey points = individual step evaluations; red = mean ± SE per shard",
      x = "Shard run",
      y = "External median error (km)"
    ) +
    theme_va() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Bootstrap CI plot for the current best model's accuracy-at-threshold
#' metrics.
plot_bootstrap_ci <- function(boot_summary) {
  boot_summary$threshold <- factor(boot_summary$threshold, levels = boot_summary$threshold)
  ggplot(boot_summary, aes(x = threshold, y = estimate)) +
    geom_col(fill = "#4C78A8", width = 0.6) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = "#1B3A5C", linewidth = 0.8) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Bootstrapped 95% CIs: current best model, external benchmark",
      subtitle = paste0("n = ", boot_summary$n[[1]], " external sample, ", boot_summary$n_boot[[1]], " bootstrap replicates"),
      x = "Accuracy threshold",
      y = "Accuracy"
    ) +
    theme_va()
}
