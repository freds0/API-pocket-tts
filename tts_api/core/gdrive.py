"""Download a public Google Drive checkpoint once, with locking and resume support."""

import logging
import zipfile
from pathlib import Path

from filelock import FileLock

from tts_api.core.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)
DEFAULT_MINIMUM_BYTES = 100_000_000


def ensure_checkpoint(
    checkpoint_path: Path,
    url: str,
    filename: str = "step500-fm0.4251.ckpt",
    minimum_bytes: int = DEFAULT_MINIMUM_BYTES,
) -> Path:
    """Return a local checkpoint, downloading the configured Drive file if absent."""
    checkpoint_path = checkpoint_path.expanduser()
    destination = checkpoint_path if checkpoint_path.suffix else checkpoint_path / filename

    if checkpoint_path.is_dir():
        try:
            return resolve_checkpoint(checkpoint_path)
        except FileNotFoundError:
            pass
    elif checkpoint_path.is_file():
        return checkpoint_path.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with FileLock(str(destination) + ".download.lock"):
        if destination.is_file():
            if destination.stat().st_size < minimum_bytes or not zipfile.is_zipfile(destination):
                raise ValueError(
                    f"Existing checkpoint looks incomplete or invalid: {destination}. "
                    "Remove it before retrying the download."
                )
            return destination.resolve()

        import gdown

        logger.info("Downloading Pocket-TTS checkpoint from Google Drive to %s", destination)
        downloaded = gdown.download(
            url=url,
            output=str(partial),
            quiet=False,
            fuzzy=True,
            resume=True,
        )
        if not downloaded or not partial.is_file():
            raise RuntimeError(
                "Google Drive did not provide the checkpoint. Check that the shared file "
                "is publicly accessible and that this host/container has network access."
            )
        if partial.stat().st_size < minimum_bytes or not zipfile.is_zipfile(partial):
            raise ValueError(
                f"Downloaded file is incomplete or is not a PyTorch checkpoint: {partial}. "
                "The partial file was kept so gdown can resume the download."
            )
        partial.replace(destination)
        logger.info("Checkpoint download complete: %s (%d bytes)", destination, destination.stat().st_size)
        return destination.resolve()
