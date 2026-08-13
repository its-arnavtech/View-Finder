from __future__ import annotations

import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.osv5m import OSV5MCombinedAdapter, _parse_shards  # noqa: E402
from checkpoint_soup import average_state_dicts  # noqa: E402
from train_batch import (  # noqa: E402
    GeoEmbeddingDataset,
    capture_l2sp_reference,
    l2sp_penalty,
    should_step_evaluate,
)


def test_l2sp_is_zero_at_anchor_and_pulls_back():
    module = nn.Linear(2, 1, bias=False)
    reference = capture_l2sp_reference(module)
    assert l2sp_penalty(module, reference).item() == pytest.approx(0.0)

    with torch.no_grad():
        module.weight.add_(torch.tensor([[1.0, -2.0]]))
    penalty = l2sp_penalty(module, reference)
    assert penalty.item() == pytest.approx(5.0)
    penalty.backward()
    assert torch.allclose(module.weight.grad, torch.tensor([[2.0, -4.0]]))


def test_checkpoint_soup_uses_normalized_parameter_wise_weights():
    states = [
        {"weight": torch.tensor([1.0, 3.0])},
        {"weight": torch.tensor([5.0, 7.0])},
    ]
    averaged, weights = average_state_dicts(states, [1.0, 3.0])
    assert weights == pytest.approx([0.25, 0.75])
    assert torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0]))

    with pytest.raises(ValueError, match="shape mismatch"):
        average_state_dicts([states[0], {"weight": torch.ones(3)}])


def test_two_phase_step_probe_schedule_switches_after_early_window():
    kwargs = {
        "step_eval_every": 10,
        "step_eval_until": 100,
        "step_eval_early_every": 1,
        "step_eval_early_until": 5,
    }
    assert [step for step in range(1, 31) if should_step_evaluate(step, **kwargs)] == [
        1, 2, 3, 4, 5, 10, 20, 30,
    ]
    assert not should_step_evaluate(110, **kwargs)


def test_cached_embedding_dataset_reads_indexed_float16_memmap(tmp_path):
    path = tmp_path / "features.f16"
    cache = np.memmap(path, mode="w+", dtype=np.float16, shape=(4, 3))
    cache[:] = np.arange(12, dtype=np.float16).reshape(4, 3)
    cache.flush()

    frame = pd.DataFrame({
        "embedding_path": [str(path), str(path)],
        "embedding_index": [3, 1],
        "embedding_count": [4, 4],
        "embedding_dim": [3, 3],
        "lat": [10.0, 20.0],
        "lon": [30.0, 40.0],
    })
    dataset = GeoEmbeddingDataset(frame)
    feature, coords = dataset[0]
    assert feature.dtype == torch.float16
    assert feature.tolist() == [9.0, 10.0, 11.0]
    assert coords.tolist() == [10.0, 30.0]


def test_combined_shard_parser_rejects_duplicates_and_bounds():
    assert _parse_shards("6,7,8,9,10") == [6, 7, 8, 9, 10]
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_shards("6,6")
    with pytest.raises(ValueError, match="out of range"):
        _parse_shards("98")


def test_combined_cache_encodes_once_and_removes_raw_files(tmp_path, monkeypatch):
    zip_path = tmp_path / "06.zip"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    ids = ["a", "b"]
    with zipfile.ZipFile(zip_path, "w") as archive:
        for index, image_id in enumerate(ids):
            path = source_dir / f"{image_id}.jpg"
            Image.new("RGB", (256, 256), color=(20 + index, 30, 40)).save(path)
            archive.write(path, arcname=f"images/{image_id}.jpg")

    adapter = OSV5MCombinedAdapter(
        shards="6",
        index_path=str(tmp_path / "unused.parquet"),
        raw_dir=str(tmp_path / "raw"),
        pool_dir=str(tmp_path / "pool"),
    )
    adapter._selected_metadata = pd.DataFrame({
        "image_id": ["osv5m-a", "osv5m-b"],
        "id": ids,
        "source_shard": [6, 6],
    })
    adapter.stats = {}
    monkeypatch.setattr(adapter, "_ensure_shard", lambda shard: str(zip_path))

    class FakeEncoder(nn.Module):
        def forward(self, images):
            means = images.mean(dim=(1, 2, 3))
            return torch.stack([means, means + 1], dim=1)

    base = {
        "image_id": ["osv5m-a", "osv5m-b"],
        "lat": [1.0, 2.0],
        "lon": [3.0, 4.0],
        "tile_id": ["x", "y"],
        "region": ["AA", "BB"],
        "dataset": ["osv5m-combined", "osv5m-combined"],
        "image_path": ["pending", "pending"],
    }
    full = pd.DataFrame(base)
    train, val = full.iloc[[0]].copy(), full.iloc[[1]].copy()
    train_cached, val_cached = adapter.cache_embeddings(
        train,
        val,
        model=SimpleNamespace(image_encoder=FakeEncoder()),
        device="cpu",
        batch_size=2,
        amp=False,
        num_workers=0,
    )

    cache_path = train_cached["embedding_path"].iloc[0]
    assert os.path.isfile(cache_path)
    assert not zip_path.exists()
    assert not (tmp_path / "pool" / "raw_06").exists()
    assert val_cached["embedding_path"].iloc[0] == cache_path
    assert adapter.stats["embedding_cache"]["raw_images_discarded"] == 2
