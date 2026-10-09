#!/usr/bin/env python
"""Pocket-TTS API: python main.py server | export."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["server", "export"])
    args = parser.parse_args()

    from tts_api.config import ConfigManager as config

    logging.basicConfig(level=config.LOG_LEVEL.upper())
    if args.command == "export":
        from tts_api.core.checkpoint import prepare_config
        from tts_api.core.gdrive import ensure_checkpoint

        checkpoint = ensure_checkpoint(
            config.POCKET_TTS_CHECKPOINT,
            config.POCKET_TTS_CHECKPOINT_URL,
            config.POCKET_TTS_CHECKPOINT_FILENAME,
        )
        print(prepare_config(
            checkpoint,
            config.POCKET_TTS_BASE_CONFIG,
            config.POCKET_TTS_TOKENIZER,
            config.POCKET_TTS_CACHE_DIR,
        ))
        return

    import uvicorn
    uvicorn.run(
        "tts_api.server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level=config.LOG_LEVEL,
        workers=1,
    )


if __name__ == "__main__":
    main()
