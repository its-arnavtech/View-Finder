"""Tests for the Discord bot's message handling.

The gateway itself is not exercised -- that needs a live token. Everything up
to it is: which messages the bot reacts to, the size guard, reply formatting,
and that a failed download produces an honest message rather than a traceback.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from viewfinder import discord_bot as bot


# --- which messages the bot reacts to ------------------------------------


@pytest.mark.parametrize("url", [
    "https://www.instagram.com/reel/Cabc123/",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.tiktok.com/@user/video/123",
    "https://x.com/user/status/123",
    "https://cdn.example.com/photo.jpg",
    "https://cdn.example.com/photo.jpg?width=800",
    "https://example.com/clip.mp4",
])
def test_media_urls_are_recognised(url):
    assert bot._looks_like_media_url(url) is True


@pytest.mark.parametrize("url", [
    "https://example.com/news/article",
    "https://github.com/its-arnavtech/View-Finder",
    "https://en.wikipedia.org/wiki/Geolocation",
])
def test_ordinary_links_are_ignored(url):
    """Reacting to every pasted link would make the bot unusable in a channel."""
    assert bot._looks_like_media_url(url) is False


# --- reply formatting ----------------------------------------------------


def _result(**extra):
    result = {
        "predictor": "geoclip",
        "prediction": {
            "latitude": 40.33532, "longitude": -123.94744,
            "country": "US", "admin1": "California", "city": "Rio Dell",
            "confidence_percent": 65.7,
        },
        "predictions": [
            {"rank": 1, "lat": 40.33532, "lon": -123.94744, "confidence": 65.7,
             "country": "US", "admin1": "California", "city": "Rio Dell"},
            {"rank": 2, "lat": 38.0, "lon": -122.7, "confidence": 45.2,
             "country": "US", "admin1": "California", "city": "Lagunitas"},
        ],
    }
    result.update(extra)
    return result


def test_reply_carries_place_coordinates_and_confidence():
    text = bot._format_reply(_result())
    assert "Rio Dell, California, US" in text
    assert "40.33532, -123.94744" in text
    assert "65.7%" in text
    assert "google.com/maps" in text


def test_reply_lists_alternatives():
    text = bot._format_reply(_result())
    assert "Alternatives:" in text
    assert "Lagunitas" in text


def test_reply_states_the_result_is_an_estimate():
    """The bot answers strangers' photos; it must not imply certainty."""
    assert "not a guarantee" in bot._format_reply(_result()).lower()


def test_abstention_is_reported_as_no_location():
    text = bot._format_reply({
        "predictor": "hybrid",
        "prediction": None,
        "visual_hypotheses": [{
            "rank": 1, "lat": 37.44, "lon": -121.87, "confidence": 12.0,
            "country": "US", "admin1": "California", "city": "Milpitas",
        }],
    })
    assert "No location accepted" in text
    assert "Milpitas" in text
    assert "manual review only" in text


def test_reply_fits_in_a_discord_message():
    """Discord rejects anything over 2000 characters."""
    crowded = _result()
    crowded["predictions"] = [
        {"rank": i, "lat": 1.0 * i, "lon": 2.0 * i, "confidence": 50.0,
         "country": "US", "admin1": "A" * 60, "city": "B" * 60}
        for i in range(1, 12)
    ]
    assert len(bot._format_reply(crowded)) < 2000


# --- handler behaviour ---------------------------------------------------


class _Attachment:
    def __init__(self, path: Path, size: int | None = None):
        self.filename = path.name
        self.size = size if size is not None else path.stat().st_size
        self._source = path

    async def save(self, target):
        shutil.copyfile(self._source, target)


class _Notice:
    def __init__(self, content):
        self.content = content

    async def edit(self, content):
        self.content = content


class _Message:
    def __init__(self, attachments=(), content="", is_bot=False):
        self.attachments = list(attachments)
        self.content = content
        self.author = type("Author", (), {"bot": is_bot})()
        self.replies: list[_Notice] = []

    async def reply(self, content, mention_author=True):
        notice = _Notice(content)
        self.replies.append(notice)
        return notice


def _client():
    """A bot instance without discord.Client.__init__, which needs a connection."""
    instance = bot.ViewFinderBot.__new__(bot.ViewFinderBot)
    instance._inference_lock = asyncio.Lock()
    return instance


def test_oversized_attachment_is_refused_before_download(tmp_path):
    sample = tmp_path / "big.jpg"
    sample.write_bytes(b"x" * 32)
    message = _Message(attachments=[
        _Attachment(sample, size=bot.MAX_ATTACHMENT_MB * 1024 * 1024 + 1)
    ])

    asyncio.run(bot.ViewFinderBot.on_message(_client(), message))
    assert len(message.replies) == 1
    assert "over the" in message.replies[0].content
    assert str(bot.MAX_ATTACHMENT_MB) in message.replies[0].content


def test_messages_from_other_bots_are_ignored(tmp_path):
    """Without this two bots in one channel can answer each other forever."""
    sample = tmp_path / "a.jpg"
    sample.write_bytes(b"x" * 32)
    message = _Message(attachments=[_Attachment(sample)], is_bot=True)
    asyncio.run(bot.ViewFinderBot.on_message(_client(), message))
    assert message.replies == []


def test_messages_without_media_are_ignored():
    message = _Message(content="hello, how are you")
    asyncio.run(bot.ViewFinderBot.on_message(_client(), message))
    assert message.replies == []


def test_non_media_attachment_is_ignored(tmp_path):
    sample = tmp_path / "notes.txt"
    sample.write_text("not an image")
    message = _Message(attachments=[_Attachment(sample)])
    asyncio.run(bot.ViewFinderBot.on_message(_client(), message))
    assert message.replies == []


# --- failure messages ----------------------------------------------------


def test_instagram_failure_names_instagram_and_suggests_a_workaround():
    text = bot.ViewFinderBot._download_failure_message(
        "https://www.instagram.com/reel/Cabc/", "empty media response"
    )
    assert "Instagram" in text
    assert "attaching it directly" in text
    assert "Traceback" not in text


def test_generic_link_failure_is_plain():
    text = bot.ViewFinderBot._download_failure_message(
        "https://youtu.be/abc", "video unavailable"
    )
    assert "Could not download that link" in text
    assert "video unavailable" in text


def test_token_is_required_and_the_error_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv(bot.TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit) as raised:
        bot.main()
    message = str(raised.value)
    assert bot.TOKEN_ENV in message
    assert ".env" in message
