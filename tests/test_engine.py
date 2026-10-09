import threading
from types import SimpleNamespace

import numpy as np
import torch

from tts_api.core.pocket_model import PocketTTSEngine


def test_disconnect_drains_model_before_restoring_parameters():
    consumed = []
    model = SimpleNamespace(temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4)

    def chunks(**kwargs):
        assert kwargs["copy_state"] is True
        for index in range(12):
            # The old request must retain its parameters through worker shutdown.
            assert model.temp == 0.2
            consumed.append(index)
            yield torch.ones(10)

    model.generate_audio_stream = chunks
    engine = PocketTTSEngine.__new__(PocketTTSEngine)
    engine.model = model
    engine._lock = threading.Lock()
    engine.device = "cpu"
    engine.voice_state = {}
    stream = engine.generate_streaming("Olá!", temperature=0.2, seed=42)
    assert next(stream).shape == (10,)
    stream.close()
    assert consumed == list(range(12))
    assert model.temp == 0.7
    assert not engine._lock.locked()


def test_seed_is_repeatable_and_restores_rng():
    def chunks(**kwargs):
        yield torch.randn(16)

    engine = PocketTTSEngine.__new__(PocketTTSEngine)
    engine.model = SimpleNamespace(
        temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=-4,
        generate_audio_stream=chunks,
    )
    engine._lock = threading.Lock()
    engine.device = "cpu"
    engine.voice_state = {}
    state = torch.random.get_rng_state().clone()
    first = list(engine.generate_streaming("Olá!", seed=42))[0]
    second = list(engine.generate_streaming("Olá!", seed=42))[0]
    assert np.array_equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
