"""A standalone Discord bot that geolocates images and videos.

Send it a photo (attachment or link) in a DM or a channel it can see, and it
replies with View Finder's prediction.

This is its **own process and its own bot identity**, deliberately separate
from the news bot. It only borrows that project's conventions:
``DISCORD_BOT_TOKEN`` read from a ``.env`` beside the project root via
python-dotenv, and a token that is never printed or logged, not even inside an
error message -- a leaked bot token lets anyone act as the bot.

One difference worth knowing: the news bot talks to the Discord REST API
directly and polls for DMs, because it only ever *pushes* a digest. This bot
has to *react* to arbitrary incoming messages, so it uses discord.py's gateway
client instead. Same token conventions, different transport.

Run it with::

    python viewfinder/discord_bot.py

It must stay running to respond. See "Discord bot" in README.md for creating
the application and inviting it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import discord

# Support both `python viewfinder/discord_bot.py` and `python -m viewfinder.discord_bot`.
try:
    from .formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from . import main as pipeline
except ImportError:  # pragma: no cover - script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from viewfinder.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
    from viewfinder import main as pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - dotenv is optional
    pass

logger = logging.getLogger("viewfinder.discord")

TOKEN_ENV = "DISCORD_BOT_TOKEN"

# Attachments above this are refused before download. Discord's own cap for
# non-Nitro uploads is 25 MB, so this mostly guards against a boosted server.
MAX_ATTACHMENT_MB = int(os.getenv("VIEWFINDER_MAX_ATTACHMENT_MB", "50"))
# Separate, larger cap for links, where a video is normal.
MAX_DOWNLOAD_MB = int(os.getenv("VIEWFINDER_MAX_DOWNLOAD_MB", "200"))
MAX_VIDEO_SECONDS = int(os.getenv("VIEWFINDER_MAX_VIDEO_SECONDS", "600"))

# Inference is slow enough that a bare reply would look like a hang.
PROCESSING_NOTICE = "Working on it - this can take up to a couple of minutes."

URL_PATTERN = re.compile(r"https?://\S+")

MEDIA_EXTENSIONS = set(IMAGE_EXTENSIONS) | set(VIDEO_EXTENSIONS)


def _looks_like_media_url(url: str) -> bool:
    """Cheap pre-filter so ordinary chat links do not trigger a download.

    Either the URL ends in a media extension, or it is from a host that serves
    video behind an opaque path. yt-dlp decides for real; this only avoids
    reacting to every link somebody pastes.
    """
    lowered = url.split("?", 1)[0].lower()
    if any(lowered.endswith(extension) for extension in MEDIA_EXTENSIONS):
        return True
    video_hosts = (
        "instagram.com", "tiktok.com", "youtube.com", "youtu.be", "twitter.com",
        "x.com", "reddit.com", "facebook.com", "vimeo.com", "twitch.tv",
        "streamable.com",
    )
    return any(host in lowered for host in video_hosts)


def _format_reply(result: dict[str, Any]) -> str:
    """Render a prediction as a Discord message, using main.py's own reducer."""
    summary = pipeline.summarize(result)

    if summary["mode"] == "abstained":
        lines = ["**No location accepted** - the evidence did not support one."]
        if summary["place"]:
            lines.append(
                f"Rejected hypothesis: {summary['place']} "
                f"({summary['latitude']:.4f}, {summary['longitude']:.4f}) - manual review only"
            )
        return "\n".join(lines)

    lines = [f"**{summary['place']}**"]
    if summary["latitude"] is not None:
        lines.append(
            f"`{summary['latitude']:.5f}, {summary['longitude']:.5f}`  "
            f"<https://www.google.com/maps?q={summary['latitude']:.5f},{summary['longitude']:.5f}>"
        )
    if summary["confidence"] is not None:
        lines.append(f"Confidence: {summary['confidence']:.1f}%")
    if summary["crop_note"]:
        lines.append(f"_{summary['crop_note']}_")

    alternatives = summary["alternatives"][:3]
    if alternatives:
        lines.append("")
        lines.append("Alternatives:")
        for alternative in alternatives:
            lines.append(
                f"  {alternative['rank']}. {alternative['place']} "
                f"({alternative['confidence']:.0f}%)"
            )

    lines.append("")
    lines.append("_Estimated from scenery. Not a guarantee._")
    return "\n".join(lines)


def _predict(path: str) -> dict[str, Any]:
    """Run the real pipeline. Blocking; callers push this to a thread."""
    result, _ = pipeline.analyze(path)
    return result


class ViewFinderBot(discord.Client):
    """Replies to any message carrying an image, a video, or a media link."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        # One GeoCLIP instance nearly fills the practical GPU envelope. Queue
        # simultaneous messages instead of loading multiple copies and turning
        # normal channel activity into an out-of-memory crash.
        self._inference_lock = asyncio.Lock()

    async def on_ready(self) -> None:
        logger.info("Connected as %s (id %s)", self.user, getattr(self.user, "id", "?"))
        print(f"View Finder bot online as {self.user}. Ctrl-C to stop.")

    async def on_message(self, message: Any) -> None:
        if message.author.bot:
            return

        attachment = self._first_media_attachment(message)
        url = None if attachment else self._first_media_url(message)
        if attachment is None and url is None:
            return

        notice = await message.reply(PROCESSING_NOTICE, mention_author=False)
        try:
            async with self._inference_lock:
                if attachment is not None:
                    reply = await self._handle_attachment(attachment)
                else:
                    reply = await self._handle_url(url)
        except Exception as exc:  # noqa: BLE001 - a crash here would look like silence
            logger.exception("Unhandled failure while processing a message")
            reply = (
                f"Something went wrong: {exc.__class__.__name__}: {exc}\n"
                "That is a bug rather than a limitation - the details are in the bot's log."
            )
        await notice.edit(content=reply[:1990])

    # -- input selection --------------------------------------------------

    def _first_media_attachment(self, message: Any) -> Any | None:
        for attachment in message.attachments:
            extension = Path(attachment.filename).suffix.lower()
            if extension in MEDIA_EXTENSIONS:
                return attachment
        return None

    def _first_media_url(self, message: Any) -> str | None:
        for match in URL_PATTERN.findall(message.content or ""):
            if _looks_like_media_url(match):
                return match.rstrip(">),.")
        return None

    # -- handlers ---------------------------------------------------------

    async def _handle_attachment(self, attachment: Any) -> str:
        if attachment.size > MAX_ATTACHMENT_MB * 1024 * 1024:
            return (
                f"That file is {attachment.size / 1024**2:.1f} MB, over the "
                f"{MAX_ATTACHMENT_MB} MB limit. Try a smaller version."
            )

        with tempfile.TemporaryDirectory(prefix="viewfinder-discord-") as workspace:
            target = Path(workspace) / Path(attachment.filename).name
            await attachment.save(target)
            if target.stat().st_size > MAX_ATTACHMENT_MB * 1024 * 1024:
                return f"That file exceeds the {MAX_ATTACHMENT_MB} MB limit."
            try:
                result = await asyncio.to_thread(_predict, str(target))
            except pipeline.UserInputError as exc:
                return f"Could not read that file: {exc}"
            except pipeline.PipelineError as exc:
                return f"Could not analyse that file: {exc}"
            return _format_reply(result)

    async def _handle_url(self, url: str) -> str:
        from viewfinder.media_source import MediaSourceError, download_media

        with tempfile.TemporaryDirectory(prefix="viewfinder-discord-") as workspace:
            try:
                resolved = await asyncio.to_thread(
                    download_media,
                    url,
                    download_dir=workspace,
                    max_duration_seconds=MAX_VIDEO_SECONDS,
                    max_download_bytes=MAX_DOWNLOAD_MB * 1024 * 1024,
                )
            except MediaSourceError as exc:
                return self._download_failure_message(url, str(exc))
            except Exception as exc:  # noqa: BLE001 - yt-dlp raises a wide range
                return self._download_failure_message(url, f"{exc.__class__.__name__}: {exc}")

            try:
                result = await asyncio.to_thread(_predict, str(resolved.local_path))
            except pipeline.UserInputError as exc:
                return f"Downloaded it, but could not read the media: {exc}"
            except pipeline.PipelineError as exc:
                return f"Downloaded it, but could not analyse it: {exc}"
            return _format_reply(result)

    @staticmethod
    def _download_failure_message(url: str, detail: str) -> str:
        """Say plainly why a link failed, especially for Instagram.

        Instagram breaks in ways that are not the user's fault and not fixable
        from here: login walls, rate limits, and extractor drift. An honest
        message beats a stack trace or silence.
        """
        lowered = url.lower()
        if "instagram.com" in lowered:
            return (
                "Could not download that Instagram link. Instagram frequently "
                "blocks anonymous downloads - the post may be private, or "
                "Instagram may be rate-limiting or requiring a login right now. "
                "This is unpredictable and often works again later.\n"
                "Downloading the video and attaching it directly works reliably.\n"
                f"Detail: {detail}"
            )
        return (
            "Could not download that link. It may be private, deleted, "
            "geo-blocked, or need a login.\n"
            f"Detail: {detail}"
        )


def build_client() -> Any:
    """Construct the client with the minimum intents this bot needs.

    `message_content` is a privileged intent and must also be enabled in the
    Discord developer portal; without it the bot sees attachments but reads
    every message body as empty, so links silently stop working.
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.dm_messages = True
    return ViewFinderBot(intents=intents)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(
            f"{TOKEN_ENV} is not set. Put it in {PROJECT_ROOT / '.env'} as\n"
            f"  {TOKEN_ENV}=your-bot-token\n"
            "See the 'Discord bot' section of README.md. This bot needs its own "
            "application, separate from any other bot you run."
        )

    client = build_client()
    try:
        # The token is passed straight through and never logged; discord.py
        # does not echo it either.
        client.run(token, log_handler=None)
    except Exception as exc:  # noqa: BLE001
        # Deliberately does not include the exception's repr, which for some
        # HTTP errors can carry the Authorization header.
        raise SystemExit(
            f"Discord connection failed ({exc.__class__.__name__}). Check that the "
            "token is correct and that the Message Content Intent is enabled in "
            "the developer portal."
        ) from None


if __name__ == "__main__":
    main()
