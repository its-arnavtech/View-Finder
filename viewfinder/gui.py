"""A small Tkinter desktop front end for the View Finder pipeline.

Pick a file, pick a predictor, press Run, read the answer -- without touching a
terminal. Launch it with::

    python viewfinder/gui.py

The pipeline itself is not reimplemented here. :func:`main.analyze` runs the
same orchestration the CLI runs, through the same argument parser, so the
predictor default, the screen-crop step and every validation message behave
identically in both front ends.

Threading
---------
Inference takes seconds to minutes, and Tk is single-threaded: doing the work
on the main thread would freeze the window. The pipeline therefore runs on a
worker thread which touches no widget at all. It only puts messages on a
:class:`queue.Queue`, and the main thread drains that queue from an ``after``
poll and does all the drawing. This is the one rule that keeps Tk stable.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, ttk

# Support both `python viewfinder/gui.py` (script, no package context) and
# `python -m viewfinder.gui` (module), matching how main.py handles the same.
try:
    from .formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from . import main as pipeline
except ImportError:  # pragma: no cover - script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from viewfinder.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from viewfinder import main as pipeline


WINDOW_TITLE = "View Finder"
PREDICTOR_CHOICES = (
    ("geoclip", "GeoCLIP  (visual model; predicts a location for any photo)"),
    ("hybrid", "Hybrid  (local EXIF/OCR/GeoNames evidence; abstains when unsure)"),
)


def _file_types() -> list[tuple[str, str]]:
    """Build filedialog filters from the pipeline's own accepted extensions.

    Derived rather than hardcoded so the picker can never drift from what the
    loader actually supports.
    """
    images = " ".join(f"*{extension}" for extension in sorted(IMAGE_EXTENSIONS))
    videos = " ".join(f"*{extension}" for extension in sorted(VIDEO_EXTENSIONS))
    return [
        ("Images and videos", f"{images} {videos}"),
        ("Images", images),
        ("Videos", videos),
        ("All files", "*.*"),
    ]


class _QueueProgress:
    """Progress sink handed to the pipeline from the worker thread.

    Satisfies the ``with progress_factory(label) as bar: bar.update(...)``
    contract that :mod:`main` uses. Every call just enqueues a message; the
    main thread is what actually moves the progress bar.
    """

    def __init__(self, label: str, sink: "queue.Queue[tuple[str, Any]]") -> None:
        self.label = label
        self._sink = sink

    def __enter__(self) -> "_QueueProgress":
        self._sink.put(("stage", self.label))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def update(self, done: int, total: int) -> None:
        self._sink.put(("progress", (self.label, int(done), int(total))))

    def close(self) -> None:
        self._sink.put(("progress_done", self.label))


class ViewFinderApp:
    """The main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.minsize(640, 560)

        self.selected_path = tk.StringVar(value="")
        self.predictor = tk.StringVar(value=pipeline.DEFAULT_PREDICTOR)
        self.status = tk.StringVar(value="Choose an image or video to begin.")

        # Worker -> UI messages. The worker never touches a widget.
        self._messages: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build()
        self._poll()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        # --- input file ---
        file_box = ttk.LabelFrame(outer, text="1. Input file", padding=10)
        file_box.pack(fill="x")
        ttk.Button(file_box, text="Choose file...", command=self.choose_file).pack(side="left")
        ttk.Label(
            file_box,
            textvariable=self.selected_path,
            wraplength=440,
            justify="left",
            foreground="#333333",
        ).pack(side="left", padx=(10, 0))

        # --- predictor ---
        mode_box = ttk.LabelFrame(outer, text="2. Predictor", padding=10)
        mode_box.pack(fill="x", pady=(10, 0))
        for value, label in PREDICTOR_CHOICES:
            ttk.Radiobutton(
                mode_box, text=label, value=value, variable=self.predictor
            ).pack(anchor="w")

        # --- run ---
        run_box = ttk.Frame(outer)
        run_box.pack(fill="x", pady=(12, 0))
        self.run_button = ttk.Button(run_box, text="Run prediction", command=self.start_run)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(run_box, mode="indeterminate", length=320)
        self.progress.pack(side="left", padx=(12, 0), fill="x", expand=True)

        ttk.Label(outer, textvariable=self.status, foreground="#444444").pack(
            anchor="w", pady=(8, 0)
        )

        # --- results ---
        result_box = ttk.LabelFrame(outer, text="Result", padding=10)
        result_box.pack(fill="both", expand=True, pady=(10, 0))
        self.results = tk.Text(
            result_box, height=16, wrap="word", state="disabled", relief="flat",
            background="#fbfbfb", font=("Consolas", 10),
        )
        scrollbar = ttk.Scrollbar(result_box, command=self.results.yview)
        self.results.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.results.pack(side="left", fill="both", expand=True)

        # Tags let errors and headings stand out without any extra dependency.
        self.results.tag_configure("heading", font=("Consolas", 11, "bold"))
        self.results.tag_configure("error", foreground="#b00020")
        self.results.tag_configure("muted", foreground="#666666")

    # -- actions ----------------------------------------------------------

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose an image or video", filetypes=_file_types())
        if path:
            self.selected_path.set(path)
            self.status.set(f"Ready: {Path(path).name}")

    def start_run(self) -> None:
        """Validate cheaply, then hand the work to a background thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        source = self.selected_path.get().strip()
        if not source:
            self._show_error("No file selected", "Choose an image or video file first.")
            return

        predictor = self.predictor.get()
        self._set_running(True)
        self._write_results([("Working...\n", "muted")])
        self.status.set(f"Running {predictor}...")

        self._worker = threading.Thread(
            target=self._run_pipeline, args=(source, predictor), daemon=True
        )
        self._worker.start()

    def _run_pipeline(self, source: str, predictor: str) -> None:
        """Worker thread. Puts results or errors on the queue; draws nothing."""

        def progress_factory(label: str) -> _QueueProgress:
            return _QueueProgress(label, self._messages)

        try:
            result, output_path = pipeline.analyze(
                source,
                predictor=predictor,
                progress_factory=progress_factory,
            )
            self._messages.put(("result", (result, output_path)))
        except (pipeline.UserInputError, pipeline.PipelineError) as exc:
            # Already phrased for a person by the CLI layer; show it verbatim.
            self._messages.put(("error", ("Could not complete the run", str(exc))))
        except Exception as exc:  # noqa: BLE001 - the UI must never die silently
            self._messages.put(
                (
                    "error",
                    (
                        f"Unexpected {exc.__class__.__name__}",
                        f"{exc}\n\nDetail:\n{traceback.format_exc(limit=3)}",
                    ),
                )
            )

    # -- main-thread message pump -----------------------------------------

    def _poll(self) -> None:
        """Drain worker messages and update widgets. Main thread only."""
        try:
            while True:
                kind, payload = self._messages.get_nowait()
                if kind == "stage":
                    self.status.set(f"{payload}...")
                elif kind == "progress":
                    self._update_progress(*payload)
                elif kind == "progress_done":
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(12)
                elif kind == "result":
                    self._set_running(False)
                    self._show_result(*payload)
                elif kind == "error":
                    self._set_running(False)
                    self._show_error(*payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _update_progress(self, label: str, done: int, total: int) -> None:
        """Switch to a real percentage once the pipeline knows the total."""
        if total > 0:
            if self.progress["mode"] != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=100)
            self.progress["value"] = min(100.0, 100.0 * done / total)
            self.status.set(f"{label}: {done}/{total}")
        else:
            self.status.set(f"{label}: {done}")

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        if running:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress["value"] = 0

    # -- rendering --------------------------------------------------------

    def _write_results(self, chunks: list[tuple[str, str]]) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        for text, tag in chunks:
            self.results.insert("end", text, tag)
        self.results.configure(state="disabled")

    def _show_error(self, title: str, detail: str) -> None:
        self.status.set("Failed.")
        self._write_results([(f"{title}\n\n", "error"), (detail + "\n", None)])

    def _show_result(self, result: dict[str, Any], output_path: Path) -> None:
        """Render the summary using the same reducer the CLI prints from."""
        summary = pipeline.summarize(result)
        chunks: list[tuple[str, str]] = []

        if summary["abstained"]:
            chunks.append(("No location accepted (fail-closed abstention)\n\n", "heading"))
        else:
            chunks.append((f"{summary['place']}\n\n", "heading"))

        if summary["latitude"] is not None:
            label = "Hypothesis" if summary["abstained"] else "Coordinates"
            chunks.append(
                (f"{label:<14}{summary['latitude']:.6f}, {summary['longitude']:.6f}\n", None)
            )
        if summary["confidence"] is not None:
            chunks.append((f"{'Confidence':<14}{summary['confidence']:.1f}%\n", None))
        chunks.append((f"{'Predictor':<14}{summary['predictor']}\n", None))
        if summary["crop_note"]:
            chunks.append((f"{'Preprocessing':<14}{summary['crop_note']}\n", "muted"))
        if summary["warning"]:
            chunks.append((f"\n! {summary['warning']}\n", "error"))

        alternatives = summary["alternatives"]
        if alternatives:
            chunks.append(("\nAlternative predictions\n", "heading"))
            for alternative in alternatives:
                chunks.append(
                    (
                        f"  {alternative['rank']}. {alternative['place']}\n"
                        f"     {alternative['confidence']:.1f}%  "
                        f"{alternative['latitude']:.6f}, {alternative['longitude']:.6f}\n",
                        None,
                    )
                )
        else:
            chunks.append(("\nNo alternative predictions.\n", "muted"))

        chunks.append((f"\nJSON written to:\n{output_path}\n", "muted"))
        self._write_results(chunks)
        self.status.set("Done.")


def main() -> None:
    """Open the window."""
    root = tk.Tk()
    ViewFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
