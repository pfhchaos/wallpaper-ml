"""Regression tests for GPU buffer/pipeline leaks in wallpaper-ml.

We had a full system lockup once because backward() was allocating new buffers
on every call without destroying old ones. These tests run training-shaped
workloads and assert that buffer/pipeline counts don't grow.

Tracks via the buffer_count / pipeline_count properties on VkCompute.
"""

import numpy as np
import pytest

try:
    from wallpaper_ml.vk_compute import VkCompute
    gpu = VkCompute()
    HAS_GPU = True
except Exception:
    HAS_GPU = False
    gpu = None

pytestmark = pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')


# ----------------------------------------------------------------
# Baseline: VkCompute tracking works
# ----------------------------------------------------------------

class TestTracking:
    def test_buffer_create_increments(self):
        before = gpu.buffer_count
        buf = gpu.create_buffer(1024)
        assert gpu.buffer_count == before + 1
        buf.destroy()
        assert gpu.buffer_count == before

    def test_buffer_bytes_tracked(self):
        before = gpu.buffer_bytes
        buf = gpu.create_buffer(4096)
        assert gpu.buffer_bytes == before + 4096
        buf.destroy()
        assert gpu.buffer_bytes == before

    def test_double_destroy_safe(self):
        before = gpu.buffer_count
        buf = gpu.create_buffer(64)
        buf.destroy()
        buf.destroy()  # should be a no-op
        assert gpu.buffer_count == before


# ----------------------------------------------------------------
# Layer-level leak tests
# ----------------------------------------------------------------

class TestLinearLeaks:
    def test_forward_no_leak(self):
        from wallpaper_ml import VkLinear
        B, I, O = 4, 16, 8
        layer = VkLinear(gpu, I, O, batch_size=B)
        layer.init_weights(np.random.default_rng(0))

        in_buf = gpu.create_buffer(B * I * 4)
        gpu.upload(in_buf, np.zeros(B * I, dtype=np.float32))

        # Warm up — first call populates the pipeline cache.
        layer.forward(in_buf, B, (I,))

        n_before = gpu.buffer_count
        p_before = gpu.pipeline_count
        for _ in range(20):
            layer.forward(in_buf, B, (I,))
        assert gpu.buffer_count == n_before, \
            f"Linear forward leaked {gpu.buffer_count - n_before} buffers in 20 calls"
        assert gpu.pipeline_count == p_before, \
            f"Linear forward leaked {gpu.pipeline_count - p_before} pipelines in 20 calls"

        in_buf.destroy()

    def test_forward_backward_no_leak(self):
        from wallpaper_ml import VkLinear
        B, I, O = 4, 16, 8
        layer = VkLinear(gpu, I, O, batch_size=B)
        layer.init_weights(np.random.default_rng(0))

        in_buf = gpu.create_buffer(B * I * 4)
        grad_buf = gpu.create_buffer(B * O * 4)
        gpu.upload(in_buf, np.zeros(B * I, dtype=np.float32))
        gpu.upload(grad_buf, np.zeros(B * O, dtype=np.float32))

        # Warm up (first backward allocates _grad_input_buf lazily)
        layer.forward(in_buf, B, (I,))
        layer.zero_grad()
        layer.backward(grad_buf, B)

        n_before = gpu.buffer_count
        p_before = gpu.pipeline_count
        for _ in range(20):
            layer.forward(in_buf, B, (I,))
            layer.zero_grad()
            layer.backward(grad_buf, B)
        assert gpu.buffer_count == n_before, \
            f"Linear fwd+bwd leaked {gpu.buffer_count - n_before} buffers in 20 iterations"
        assert gpu.pipeline_count == p_before, \
            f"Linear fwd+bwd leaked {gpu.pipeline_count - p_before} pipelines"

        in_buf.destroy()
        grad_buf.destroy()


class TestGRULeaks:
    def test_forward_no_leak(self):
        from wallpaper_ml import VkGRU
        B, I, H = 2, 8, 4
        layer = VkGRU(gpu, I, H, batch_size=B, max_seq_len=4)
        layer.init_weights(np.random.default_rng(0))

        in_buf = gpu.create_buffer(B * I * 4)
        gpu.upload(in_buf, np.zeros(B * I, dtype=np.float32))

        # Warm up
        layer.reset_hidden()
        for _ in range(4):
            layer.forward(in_buf, B, (I,))

        n_before = gpu.buffer_count
        p_before = gpu.pipeline_count
        for _ in range(20):
            layer.reset_hidden()
            for _ in range(4):
                layer.forward(in_buf, B, (I,))
        assert gpu.buffer_count == n_before, \
            f"GRU forward leaked {gpu.buffer_count - n_before} buffers"
        assert gpu.pipeline_count == p_before, \
            f"GRU forward leaked {gpu.pipeline_count - p_before} pipelines"

        in_buf.destroy()

    def test_forward_backward_no_leak(self):
        from wallpaper_ml import VkGRU
        B, I, H = 2, 8, 4
        layer = VkGRU(gpu, I, H, batch_size=B, max_seq_len=4)
        layer.init_weights(np.random.default_rng(0))

        in_buf = gpu.create_buffer(B * I * 4)
        grad_buf = gpu.create_buffer(B * H * 4)
        gpu.upload(in_buf, np.zeros(B * I, dtype=np.float32))
        gpu.upload(grad_buf, np.zeros(B * H, dtype=np.float32))

        # Warm up
        layer.reset_hidden()
        for _ in range(4):
            layer.forward(in_buf, B, (I,))
        layer.zero_grad()
        layer.backward(grad_buf, B)

        n_before = gpu.buffer_count
        p_before = gpu.pipeline_count
        for _ in range(20):
            layer.reset_hidden()
            for _ in range(4):
                layer.forward(in_buf, B, (I,))
            layer.zero_grad()
            layer.backward(grad_buf, B)
        assert gpu.buffer_count == n_before, \
            f"GRU fwd+bwd leaked {gpu.buffer_count - n_before} buffers in 20 iterations"
        assert gpu.pipeline_count == p_before, \
            f"GRU fwd+bwd leaked {gpu.pipeline_count - p_before} pipelines"

        in_buf.destroy()
        grad_buf.destroy()


class TestBeatCRNNLeaks:
    """End-to-end model training loop must not leak."""

    def test_training_step_no_leak(self):
        from wallpaper_ml import build_beat_crnn, VkGRU

        B = 2
        I = 216
        T = 8  # short chunk
        model = build_beat_crnn(gpu, input_size=I, hidden_size=8,
                                n_classes=3, batch_size=B, max_seq_len=T)
        model.init_weights(seed=0)

        in_buf = gpu.create_buffer(B * I * 4)
        grad_buf = gpu.create_buffer(B * 3 * 4)

        def one_step():
            model.zero_grad()
            for layer in model.layers:
                if isinstance(layer, VkGRU):
                    layer.reset_hidden()
            for _ in range(T):
                gpu.upload(in_buf, np.random.randn(B * I).astype(np.float32))
                model.forward(in_buf, B, (I,))
            gpu.upload(grad_buf, np.random.randn(B * 3).astype(np.float32))
            model.backward(grad_buf, B)
            model.sgd_step(0.001)

        # Warm up: lets layers lazily allocate any internal buffers
        one_step()

        n_before = gpu.buffer_count
        b_before = gpu.buffer_bytes
        p_before = gpu.pipeline_count

        # Now run many "training steps" — counts must stay flat
        for _ in range(10):
            one_step()

        assert gpu.buffer_count == n_before, \
            f"BeatCRNN leaked {gpu.buffer_count - n_before} buffers " \
            f"({gpu.buffer_bytes - b_before} bytes) over 10 training steps"
        assert gpu.pipeline_count == p_before, \
            f"BeatCRNN leaked {gpu.pipeline_count - p_before} pipelines"

        in_buf.destroy()
        grad_buf.destroy()


class TestCNNScorerLeaks:
    """End-to-end CNN scorer training step — the exact path that locks up
    the GPU during long training runs."""

    def test_full_training_step_no_leak(self):
        from wallpaper_ml import build_cnn_scorer

        B = 2
        layers_config = [
            (4, 8, 3, 1, 1),
            (8, 16, 3, 2, 1),
            (16, 16, 3, 1, 1),
        ]
        model = build_cnn_scorer(gpu, layers_config, batch_size=B,
                                  image_size=64, mlp_head=False)
        model.init_weights(seed=0)

        in_buf = gpu.create_buffer(B * 4 * 64 * 64 * 4)
        grad_buf = gpu.create_buffer(B * 4)
        gpu.upload(in_buf, np.random.randn(B * 4 * 64 * 64).astype(np.float32))
        gpu.upload(grad_buf, np.random.randn(B).astype(np.float32))

        def one_step():
            model.zero_grad()
            model.forward(in_buf, B, (4, 64, 64))
            model.backward(grad_buf, B)
            model.sgd_step(0.001)

        one_step()  # warm up — lazy buffer allocation, pipeline cache

        n_before = gpu.buffer_count
        b_before = gpu.buffer_bytes
        p_before = gpu.pipeline_count

        for _ in range(10):
            one_step()

        assert gpu.buffer_count == n_before, \
            f"CNN training leaked {gpu.buffer_count - n_before} buffers " \
            f"({(gpu.buffer_bytes - b_before) / 1e6:.1f} MB) over 10 steps"
        assert gpu.pipeline_count == p_before, \
            f"CNN training leaked {gpu.pipeline_count - p_before} pipelines over 10 steps"

        in_buf.destroy()
        grad_buf.destroy()

    def test_pairwise_training_step_no_leak(self):
        """Pairwise (winner/loser re-forward + double backward) — what the
        finetune script does."""
        from wallpaper_ml import build_cnn_scorer

        B = 2
        layers_config = [
            (4, 8, 3, 1, 1),
            (8, 16, 3, 2, 1),
        ]
        model = build_cnn_scorer(gpu, layers_config, batch_size=B,
                                  image_size=64, mlp_head=False)
        model.init_weights(seed=0)

        in_buf = gpu.create_buffer(B * 4 * 64 * 64 * 4)
        d_scores_buf = gpu.create_buffer(B * 4)
        winner = np.random.randn(B * 4 * 64 * 64).astype(np.float32)
        loser = np.random.randn(B * 4 * 64 * 64).astype(np.float32)
        d_loser = np.full(B, 1.0 / B, dtype=np.float32)
        d_winner = np.full(B, -1.0 / B, dtype=np.float32)

        def pairwise_step():
            model.zero_grad()
            gpu.upload(in_buf, winner)
            model.forward(in_buf, B, (4, 64, 64))
            gpu.upload(in_buf, loser)
            model.forward(in_buf, B, (4, 64, 64))
            gpu.upload(d_scores_buf, d_loser)
            model.backward(d_scores_buf, B)
            gpu.upload(in_buf, winner)
            model.forward(in_buf, B, (4, 64, 64))
            gpu.upload(d_scores_buf, d_winner)
            model.backward(d_scores_buf, B)
            model.sgd_step(0.001)

        pairwise_step()  # warm up

        n_before = gpu.buffer_count
        b_before = gpu.buffer_bytes
        p_before = gpu.pipeline_count

        for _ in range(10):
            pairwise_step()

        assert gpu.buffer_count == n_before, \
            f"Pairwise training leaked {gpu.buffer_count - n_before} buffers " \
            f"({(gpu.buffer_bytes - b_before) / 1e6:.1f} MB) over 10 steps"
        assert gpu.pipeline_count == p_before, \
            f"Pairwise training leaked {gpu.pipeline_count - p_before} pipelines over 10 steps"

        in_buf.destroy()
        d_scores_buf.destroy()
