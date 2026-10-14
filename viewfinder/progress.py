"""Progress reporting shared by the CLI and the desktop GUI.

Deliberately dependency-free. tqdm is present in the environment as a
transitive dependency of transformers and open-clip, but it is not a declared
View Finder requirement and pulling it in would only solve half the problem:
the GUI needs the same progress events delivered to a Tk widget, not written to
a terminal. One small abstraction serves both.

The pieces
----------
``ProgressCallback``
    ``(done, total) -> None``. This is what the predictors emit.

:class:`TerminalProgressBar`
    Renders a carriage-return bar for the CLI. Silent unless the stream is a
    real terminal, so redirected output and log files never collect control
    characters.

:func:`report`
    Bridges the predictors' historical ``progress: bool`` parameter to the new
    callback form without breaking either. ``progress=True`` keeps logging a
    line per batch exactly as before; ``progress=<callable>`` drives a bar.
"""

from __future__ import annotations

import logging
import sys
from types import TracebackType
from typing import Any, Callable, Protocol, TextIO


# (completed, total) -> None
ProgressCallback = Callable[[int, int], None]

_BAR_WIDTH = 28
_FILLED = "#"
_EMPTY = "."


class SupportsProgress(Protocol):
    """What both the terminal bar and the GUI adapter provide."""

    def update(self, done: int, total: int) -> None: ...

    def close(self) -> None: ...


class TerminalProgressBar:
    """A single-line carriage-return progress bar.

    Writes to stderr so a ``--quiet`` run's JSON on stdout stays parseable.
    Rendering is suppressed when the stream is not a TTY (piped output, CI
    logs) or when ``enabled`` is False, in which case every method is a no-op
    and the caller needs no special case.
    """

    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        enabled: bool = True,
        width: int = _BAR_WIDTH,
    ) -> None:
        self.label = label
        self.width = width
        self._stream = stream if stream is not None else sys.stderr
        self._enabled = enabled and self._is_tty(self._stream)
        self._closed = False
        self._last_line = ""

    @staticmethod
    def _is_tty(stream: TextIO) -> bool:
        """True only for real terminals. Never raises on odd stream objects."""
        try:
            return bool(stream.isatty())
        except Exception:  # pragma: no cover - exotic stream wrappers
            return False

    def update(self, done: int, total: int) -> None:
        """Redraw the bar in place. Values are clamped so bad input can't corrupt it."""
        if not self._enabled or self._closed:
            return
        total = max(int(total), 0)
        done = min(max(int(done), 0), total) if total else max(int(done), 0)
        if total:
            fraction = done / total
            filled = int(round(fraction * self.width))
            bar = _FILLED * filled + _EMPTY * (self.width - filled)
            line = f"  {self.label} [{bar}] {done}/{total} ({fraction * 100:5.1f}%)"
        else:
            # Unknown total: report the count alone rather than a fake ratio.
            line = f"  {self.label} [{_EMPTY * self.width}] {done}"
        # Pad to erase the tail of a previously longer line.
        padding = " " * max(len(self._last_line) - len(line), 0)
        self._write(f"\r{line}{padding}")
        self._last_line = line

    def close(self) -> None:
        """Finish the line so following output starts cleanly. Idempotent."""
        if not self._enabled or self._closed:
            self._closed = True
            return
        self._closed = True
        if self._last_line:
            self._write("\n")

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:  # pragma: no cover - closed/detached stream
            self._enabled = False

    # Context-manager use guarantees the line is terminated even on error, so a
    # traceback or an error message never lands mid-bar.
    def __enter__(self) -> "TerminalProgressBar":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def callback(self) -> ProgressCallback:
        """Return this bar as a plain ``(done, total)`` callable."""
        return self.update


def report(
    progress: Any,
    done: int,
    total: int,
    *,
    logger: logging.Logger,
    message: str,
) -> None:
    """Emit one progress tick for whichever ``progress`` form the caller passed.

    Args:
        progress: ``False``/``None`` to stay silent, ``True`` for the legacy
            log line, or a ``(done, total)`` callable for a bar or GUI widget.
        done: Units completed so far.
        total: Total units.
        logger: Logger used for the ``progress=True`` form.
        message: printf-style template taking ``(done, total)``.
    """
    if not progress:
        return
    if callable(progress):
        try:
            progress(done, total)
        except Exception:  # pragma: no cover - a broken UI must not kill a run
            logger.debug("Progress callback raised; continuing", exc_info=True)
        return
    logger.info(message, done, total)
