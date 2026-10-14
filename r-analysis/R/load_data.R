#' Data loading for the R statistical analysis layer.
#'
#' Reads directly from finetune/work/*/summary.json — the same files the
#' Python training pipeline already writes after every run. Nothing here
#' modifies or depends on any change to how those files are produced, so
#' new runs are picked up automatically just by existing on disk.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(stringr)
})

#' Locate the finetune/work directory relative to this project.
#' Walks up from the given start directory looking for a "finetune/work"
#' sibling, so the script works whether it's run from r-analysis/ or from
#' the repo root.
locate_finetune_work <- function(start_dir = getwd()) {
  candidates <- c(
    file.path(start_dir, "finetune", "work"),
    file.path(start_dir, "..", "finetune", "work"),
    file.path(start_dir, "..", "..", "finetune", "work")
  )
  hit <- candidates[dir.exists(candidates)]
  if (length(hit) == 0) {
    stop(
      "Could not find finetune/work/ from ", start_dir,
      ". Run this script from r-analysis/ or the repo root."
    )
  }
  normalizePath(hit[[1]])
}

#' Pull the run_name a checkpoint path was produced by, e.g.
#' ".../checkpoints/osv5m04/location_encoder_best_external.pth" -> "osv5m04"
checkpoint_run_name <- function(path) {
  if (is.null(path) || length(path) == 0 || is.na(path)) return(NA_character_)
  m <- str_match(path, "checkpoints[\\\\/]+([^\\\\/]+)[\\\\/]")
  m[, 2]
}

#' Parse dataset_arg entries like "shard=0" or "shards=6,7,8" into a
#' normalized comma-separated shard id string, e.g. "0" or "6,7,8".
parse_shards <- function(dataset_arg) {
  if (is.null(dataset_arg) || length(dataset_arg) == 0) return(NA_character_)
  hit <- dataset_arg[str_detect(dataset_arg, "^shards?=")]
  if (length(hit) == 0) return(NA_character_)
  str_remove(hit[[1]], "^shards?=")
}

#' Safely pluck a possibly-absent nested value, returning NA instead of
#' erroring. `path` is a character vector of successive list names.
pluck_num <- function(x, path, default = NA_real_) {
  v <- purrr::pluck(x, !!!as.list(path), .default = default)
  if (is.null(v) || length(v) == 0) return(default)
  suppressWarnings(as.numeric(v))
}
pluck_chr <- function(x, path, default = NA_character_) {
  v <- purrr::pluck(x, !!!as.list(path), .default = default)
  if (is.null(v) || length(v) == 0) return(default)
  as.character(v)
}

#' Find the selector entry (from the `selectors` list) with a given name,
#' e.g. "external" or "indist".
find_selector <- function(js, name) {
  sels <- js$selectors
  if (is.null(sels) || length(sels) == 0) return(NULL)
  # jsonlite simplifies `selectors` to a data.frame when the shape is
  # regular, or a list of lists otherwise -- handle both.
  if (is.data.frame(sels)) {
    row <- sels[sels$name == name, , drop = FALSE]
    if (nrow(row) == 0) return(NULL)
    as.list(row[1, ])
  } else {
    hit <- purrr::keep(sels, ~ identical(.x$name, name))
    if (length(hit) == 0) return(NULL)
    hit[[1]]
  }
}

#' Parse a single summary.json file into one flat row of the columns the
#' analyses need, plus a list-column carrying the full external_trend
#' (step-level eval history) for the variance-decomposition analysis.
parse_run_summary <- function(path) {
  js <- tryCatch(
    jsonlite::fromJSON(path, simplifyVector = TRUE),
    error = function(e) NULL
  )
  if (is.null(js)) {
    warning("Skipping unparseable summary.json: ", path)
    return(NULL)
  }

  sel_ext <- find_selector(js, "external")
  sel_indist <- find_selector(js, "indist")

  # selected_checkpoints$external$external holds the accuracy/error metrics
  # of the checkpoint that was chosen by the external-benchmark selector,
  # evaluated on the external benchmark itself.
  ext_metrics <- purrr::pluck(js, "selected_checkpoints", "external", "external", .default = list())

  reg <- js$config$reg_strength
  reg_strength <- if (is.null(reg)) NA_real_ else as.numeric(reg)
  anchor_path <- purrr::pluck(js, "regularization", "anchor", .default = js$config$init_checkpoint)

  trend <- js$external_trend
  if (!is.data.frame(trend)) trend <- tryCatch(as.data.frame(trend), error = function(e) NULL)

  tibble::tibble(
    run_name = js$run_name %||% basename(dirname(path)),
    dataset = js$config$dataset %||% NA_character_,
    shards = parse_shards(js$config$dataset_arg),
    reg_strength = reg_strength,
    anchor_run = checkpoint_run_name(anchor_path),
    external_best_median_km = if (is.null(sel_ext)) NA_real_ else as.numeric(sel_ext$best_value),
    external_best_epoch = if (is.null(sel_ext)) NA_real_ else as.numeric(sel_ext$best_epoch),
    external_best_step = if (is.null(sel_ext)) NA_real_ else as.numeric(sel_ext$best_step),
    indist_best_median_km = if (is.null(sel_indist)) NA_real_ else as.numeric(sel_indist$best_value),
    ext_sel_n = pluck_num(ext_metrics, "n"),
    ext_sel_median_km = pluck_num(ext_metrics, "median_error_km"),
    ext_sel_mean_km = pluck_num(ext_metrics, "mean_error_km"),
    ext_sel_acc_25km = pluck_num(ext_metrics, "acc_25km"),
    ext_sel_acc_200km = pluck_num(ext_metrics, "acc_200km"),
    ext_sel_acc_750km = pluck_num(ext_metrics, "acc_750km"),
    ext_sel_acc_2500km = pluck_num(ext_metrics, "acc_2500km"),
    external_baseline_median_km = pluck_num(js, c("external_baseline", "median_error_km")),
    source_file = path,
    mtime = file.info(path)$mtime,
    external_trend = list(trend)
  )
}

#' Discover and load every run in finetune/work/. Resilient to new run
#' directories appearing over time -- just re-list the directory.
#' Runs whose summary.json fails to parse are dropped with a warning
#' rather than aborting the whole analysis.
load_all_runs <- function(work_dir = locate_finetune_work()) {
  summary_files <- list.files(
    work_dir, pattern = "^summary\\.json$",
    recursive = TRUE, full.names = TRUE
  )
  if (length(summary_files) == 0) {
    stop("No summary.json files found under ", work_dir)
  }
  rows <- purrr::map(summary_files, parse_run_summary)
  rows <- purrr::compact(rows)
  dplyr::bind_rows(rows) |> dplyr::arrange(run_name)
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || (length(a) == 1 && is.na(a))) b else a
