from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

from tts_api.core.checkpoint import prepare_config, resolve_checkpoint


def test_last_checkpoint_takes_precedence(tmp_path):
    old = tmp_path / "step50-fm0.1.ckpt"
    newest = tmp_path / "step500-fm0.2.ckpt"
    old.touch()
    newest.touch()
    (tmp_path / "last.ckpt").symlink_to(old.name)
    assert resolve_checkpoint(tmp_path) == old


def test_fallback_uses_numeric_step(tmp_path):
    for step in (9, 50, 500):
        (tmp_path / f"step{step}-fm0.1.ckpt").touch()
    assert resolve_checkpoint(tmp_path).name == "step500-fm0.1.ckpt"


def test_missing_checkpoint_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)


def test_export_uses_trained_weights_and_local_tokenizer(tmp_path):
    checkpoint = tmp_path / "step500.ckpt"
    torch.save({"state_dict": {
        "flow_lm.test": torch.ones(2, dtype=torch.bfloat16),
        "mimi.test": torch.zeros(3),
        "training_metric": torch.ones(1),
    }}, checkpoint)
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"local-tokenizer")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "weights_path": "old-weights",
        "flow_lm": {"lookup_table": {"tokenizer_path": "remote-tokenizer"}, "weights_path": "old-flow"},
        "mimi": {"sample_rate": 24000, "weights_path": "old-mimi"},
    }))
    output = prepare_config(checkpoint, base, tokenizer, tmp_path / "cache")
    config = yaml.safe_load(output.read_text())
    tensors = load_file(config["weights_path"])
    assert set(tensors) == {"flow_lm.test", "mimi.test"}
    assert tensors["flow_lm.test"].dtype == torch.float32
    assert config["flow_lm"]["lookup_table"]["tokenizer_path"] == str(tokenizer)
    assert "weights_path" not in config["mimi"]
    original_mtime = output.stat().st_mtime_ns
    assert prepare_config(checkpoint, base, tokenizer, tmp_path / "cache") == output
    assert output.stat().st_mtime_ns == original_mtime
    tokenizer.write_bytes(b"different-tokenizer")
    assert prepare_config(checkpoint, base, tokenizer, tmp_path / "cache") != output
