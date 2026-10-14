"""Fine-tuned GeoCLIP is the product default; stock is explicit."""

from __future__ import annotations

from pathlib import Path

import pytest

from viewfinder import geolocator, main


def test_bundled_finetuned_checkpoint_resolves() -> None:
    checkpoint = geolocator._resolve_location_encoder(None)
    assert checkpoint.is_file()
    assert checkpoint.name == "location_encoder_best_external.pth"
    assert "osv5m_reg_narrow_30" in str(checkpoint)


def test_missing_explicit_checkpoint_does_not_silently_use_stock(tmp_path) -> None:
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError, match="--use-stock-geoclip"):
        geolocator._resolve_location_encoder(missing)


def test_geolocator_factory_defaults_to_finetuned_and_forwards_stock(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeGeoLocator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(main, "_import_attr", lambda module, attribute: FakeGeoLocator)

    main._new_geolocator()
    main._new_geolocator(use_stock_geoclip=True)

    assert calls == [
        {"offline": True, "use_stock_geoclip": False},
        {"offline": True, "use_stock_geoclip": True},
    ]


def test_stock_flag_is_available_to_cli() -> None:
    args = main.build_parser().parse_args(["photo.jpg", "--use-stock-geoclip"])
    assert args.use_stock_geoclip is True
