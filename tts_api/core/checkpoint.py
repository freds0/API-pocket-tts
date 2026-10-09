"""Export the trusted local training checkpoint for the stock inference loader."""

import hashlib
import json
import re
from pathlib import Path

import yaml
from filelock import FileLock


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    last = path / "last.ckpt"
    if last.is_file():
        return last.resolve()
    candidates = list(path.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No last.ckpt or *.ckpt files found in {path}")

    def rank(candidate: Path):
        match = re.search(r"step[=_-]?(\d+)", candidate.name)
        return (int(match[1]) if match else -1, candidate.stat().st_mtime_ns, candidate.name)

    return max(candidates, key=rank).resolve()


def prepare_config(checkpoint: Path, base_config: Path, tokenizer: Path, cache_dir: Path) -> Path:
    checkpoint = resolve_checkpoint(checkpoint)
    base_config = base_config.resolve(strict=True)
    tokenizer = tokenizer.resolve(strict=True)
    source = base_config.read_bytes()
    stat = checkpoint.stat()
    identity = {
        "version": 1,
        "checkpoint": str(checkpoint),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config_sha256": hashlib.sha256(source).hexdigest(),
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    out_dir = cache_dir.resolve() / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    config_file = out_dir / "config.yaml"
    with FileLock(str(out_dir / "export.lock")):
        if config_file.is_file():
            return config_file
        weights = checkpoint
        if checkpoint.suffix != ".safetensors":
            import torch
            from safetensors.torch import save_file

            # Only administrator-configured checkpoints are loaded. Lightning
            # metadata requires pickle; request bodies cannot select a path.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("state_dict", payload)
            tensors = {
                key: value.detach().float().contiguous().clone()
                for key, value in state.items()
                if key.startswith(("flow_lm.", "mimi."))
            }
            if not tensors or not any(key.startswith("mimi.") for key in tensors):
                raise ValueError("Checkpoint must contain flow_lm.* and mimi.* weights")
            if not any(key.startswith("flow_lm.") for key in tensors):
                raise ValueError("Checkpoint does not contain flow_lm.* weights")
            weights = out_dir / "model.safetensors"
            temporary = out_dir / "model.safetensors.tmp"
            save_file(tensors, str(temporary))
            temporary.replace(weights)
            del tensors, state, payload
        config = yaml.safe_load(source)
        config["weights_path"] = str(weights)
        config["weights_path_without_voice_cloning"] = str(weights)
        config["flow_lm"].pop("weights_path", None)
        config["mimi"].pop("weights_path", None)
        config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
        (out_dir / "source.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        temporary_config = out_dir / "config.yaml.tmp"
        temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        temporary_config.replace(config_file)
    return config_file
