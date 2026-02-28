"""
Ensure LiveBufferManager processes signals even if start() is not called.
"""

import torch

from dual_stream.encoder import LiveStateEncoder, LiveBufferManager


class _TinyEmbedder:
    def __call__(self, text: str) -> torch.Tensor:
        # Deterministic tiny embedding
        torch.manual_seed(0)
        return torch.ones(1, 8)


def test_live_buffer_processes_without_start():
    encoder = LiveStateEncoder(hidden_dim=16, input_dim=8, num_signal_slots=2)
    manager = LiveBufferManager(encoder, text_embedder=_TinyEmbedder())

    # Post a signal without explicitly starting the worker
    manager.post_signal("alert: test", priority=0.7)

    state = manager.get_current_state()

    assert state is not None, "Live state should be produced even without start()"
    assert state.shape[-1] == 16
    assert abs(manager.get_max_priority() - 0.7) < 1e-4



