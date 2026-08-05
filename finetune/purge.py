"""Discard a batch's raw image data after training.

The batch-ingest-train-discard strategy assumes the raw images are disposable
scratch: pulled for one batch, encoded, trained on, deleted, with the location
encoder checkpoint as the only artifact that persists.

That assumption holds for *freshly staged* batches. It does NOT hold when a
batch is assembled by reference from an existing dataset. The first run reuses
gallery-pipeline data, which is committed source data that also backs the FAISS
retrieval galleries -- metadata.parquet stores raw_file_path pointers into those
same image files, so deleting them would silently break the gallery side of
View-Finder and destroy work that is not reproducible without re-scraping.

So purging is deliberately narrow: it deletes only files that live inside the
run's own staging directory, and refuses anything outside it. A batch built by
reference stages no copies, so purging it is correctly a no-op.
"""

from __future__ import annotations

import os
import shutil

WORK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
STAGED_SUBDIR = "staged_images"


def _is_inside(path: str, root: str) -> bool:
    """True only if `path` resolves to a location strictly inside `root`."""
    path_real = os.path.realpath(path)
    root_real = os.path.realpath(root)
    return os.path.commonpath([path_real, root_real]) == root_real and path_real != root_real


def purge_batch(work_dir: str, frame) -> dict:
    """Delete this batch's staged raw images. Never touches source datasets.

    Returns a small report describing what was and was not deleted.
    """
    staged_dir = os.path.join(work_dir, STAGED_SUBDIR)

    # Hard guard: the staging directory must live under the finetune work root.
    if not _is_inside(staged_dir, WORK_ROOT):
        raise RuntimeError(
            f"Refusing to purge: staging dir {staged_dir!r} is not inside the work root {WORK_ROOT!r}."
        )

    referenced_outside = 0
    if frame is not None and "image_path" in frame:
        for path in frame["image_path"]:
            if not _is_inside(str(path), staged_dir):
                referenced_outside += 1

    if not os.path.isdir(staged_dir):
        print(
            f"\n[purge] Nothing to delete: this batch was assembled by reference, not staged as copies.\n"
            f"[purge] {referenced_outside} training images live in source datasets outside the run's\n"
            f"[purge] staging area and were deliberately NOT deleted -- gallery-pipeline raw images are\n"
            f"[purge] committed source data that the FAISS galleries still point at.\n"
            f"[purge] Checkpoints under finetune/checkpoints/ are the artifact that persists."
        )
        return {"deleted_files": 0, "deleted_bytes": 0, "skipped_source_refs": referenced_outside}

    deleted_files = 0
    deleted_bytes = 0
    for root, _dirs, files in os.walk(staged_dir):
        for name in files:
            full = os.path.join(root, name)
            if not _is_inside(full, staged_dir):
                continue  # defensive: skip anything that escaped via a link
            deleted_bytes += os.path.getsize(full)
            deleted_files += 1
    shutil.rmtree(staged_dir)

    print(
        f"\n[purge] Deleted {deleted_files} staged images "
        f"({deleted_bytes / 1024**2:.1f} MB) from {staged_dir}."
    )
    if referenced_outside:
        print(f"[purge] Left {referenced_outside} source-dataset images untouched.")
    return {
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "skipped_source_refs": referenced_outside,
    }
