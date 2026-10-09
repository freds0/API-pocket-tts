"""Download the API checkpoint from the configured Google Drive link."""

import argparse
import logging
from pathlib import Path

from tts_api.config import ConfigManager
from tts_api.core.gdrive import ensure_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ConfigManager.POCKET_TTS_CHECKPOINT)
    parser.add_argument("--url", default=ConfigManager.POCKET_TTS_CHECKPOINT_URL)
    parser.add_argument("--filename", default=ConfigManager.POCKET_TTS_CHECKPOINT_FILENAME)
    args = parser.parse_args()
    logging.basicConfig(level=ConfigManager.LOG_LEVEL.upper())
    print(ensure_checkpoint(args.checkpoint, args.url, args.filename))


if __name__ == "__main__":
    main()
