from pathlib import Path

import numpy as np

from src.data import K, SEED
from src.models import build_cnn
from src import research


def test_research_cnn_and_resumable_run(tmp_path: Path) -> None:
    """La base reprend C0, les variantes se construisent et un run déjà fait est relu."""
    base = research.build_research_cnn(research.BASE_CONFIG, SEED)
    assert base.count_params() == build_cnn().count_params() == 683527
    variant = research.build_research_cnn(
        {**research.BASE_CONFIG, "batch_norm": True, "convs_per_block": 2, "conv_dropout": 0.25, "dense_dropout": 0.5},
        SEED,
    )
    assert variant.output_shape == (None, K)

    rng = np.random.default_rng(SEED)
    X = rng.random((70, 48, 48, 1), dtype=np.float32)
    y = np.arange(70, dtype=np.int64) % K
    config = {**research.BASE_CONFIG, "max_epochs": 2, "batch_size": 16, "class_weight": True, "reduce_lr": True}
    first = research.run_research("T", "test", config, SEED, (X[:56], y[:56], X[56:], y[56:]),
                                  out_dir=tmp_path, checkpoint_dir=tmp_path, verbose=0)
    assert (tmp_path / "T_s42.json").is_file() and (tmp_path / "T_s42.keras").is_file()
    assert first["test_used"] is False and 1 <= first["best_epoch"] <= first["epochs_ran"] <= 2
    assert set(first["val_f1"]) and np.isfinite(first["val_macro_f1"])
    again = research.run_research("T", "test", config, SEED, None, out_dir=tmp_path, checkpoint_dir=tmp_path)
    assert again == first
    summary = research.summarize(tmp_path)
    assert summary.loc[0, "id"] == "T" and summary.loc[0, "seeds"] == 1
