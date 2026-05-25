"""Sequential VkModel + builders for the CNN scorer and beat-RNN."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .vk_compute import VkCompute, VkBuffer
from .layers import (
    VkLayer,
    VkConv2d,
    VkGAPLinear,
    VkGAPMLP,
    VkLinear,
    VkGRU,
)

SHADER_DIR = Path(__file__).parent / 'shaders' / 'cnn'


class VkModel:
    """Sequential model of VkLayers."""

    def __init__(self, gpu: VkCompute, layers: list[VkLayer]):
        self.gpu = gpu
        self.layers = layers
        self._sgd_pipeline = None

    def forward(self, input_buf: VkBuffer, batch_size: int,
                input_shape: tuple[int, ...]) -> VkBuffer:
        """Run forward pass through all layers. Returns output buffer."""
        x = input_buf
        shape = input_shape
        for layer in self.layers:
            x, shape = layer.forward(x, batch_size, shape)
        return x

    def backward(self, grad_output: VkBuffer, batch_size: int) -> None:
        """Run backward pass, accumulating weight gradients."""
        g = grad_output
        for layer in reversed(self.layers):
            g = layer.backward(g, batch_size)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def sgd_step(self, lr: float) -> None:
        """SGD weight update: w -= lr * grad, then zero grads.

        Layers may transform their gradient buffer before the update via
        `pre_sgd_step()` (e.g., VkConv2d uses this for per-input-channel LR
        multipliers in staged-training curricula).
        """
        for layer in self.layers:
            layer.pre_sgd_step()
            for param_buf, grad_buf, n in layer.parameters():
                pipeline = self.gpu.create_pipeline(
                    str(SHADER_DIR / 'sgd_update.comp'),
                    buffers=[param_buf, grad_buf],
                    push_constant_size=8,
                )
                push = struct.pack('fI', lr, n)
                self.gpu.dispatch(pipeline, (n + 255) // 256, push_constants=push)
                pipeline.destroy()

    def param_count(self) -> int:
        return sum(l.param_count() for l in self.layers)

    def save_weights(self) -> np.ndarray:
        """Save all weights as a flat float32 array (compatible with .npy)."""
        return np.concatenate([l.save_weights() for l in self.layers])

    def load_weights(self, flat: np.ndarray) -> None:
        """Load weights from flat float32 array."""
        offset = 0
        for layer in self.layers:
            offset = layer.load_weights(flat, offset)
        if offset != len(flat):
            raise ValueError(f'Weight count mismatch: loaded {offset}, '
                             f'file has {len(flat)}')

    def init_weights(self, seed: int = 42) -> None:
        """Initialize all weights with Kaiming uniform."""
        rng = np.random.default_rng(seed)
        for layer in self.layers:
            layer.init_weights(rng)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_cnn_scorer(gpu: VkCompute, layers_config: list[tuple],
                     batch_size: int, image_size: int = 256,
                     mlp_head: bool = False, hidden: int = 16) -> VkModel:
    """Build CNN aesthetic scorer model.

    Args:
        gpu: VkCompute instance
        layers_config: [(in_c, out_c, kernel, stride, padding), ...]
        batch_size: fixed batch size for buffer allocation
        image_size: input image H=W
        mlp_head: use MLP head instead of linear
        hidden: MLP hidden dimension
    """
    layers: list[VkLayer] = []
    h, w = image_size, image_size
    for c_in, c_out, k, s, p in layers_config:
        layers.append(VkConv2d(gpu, c_in, c_out, k, s, p,
                               batch_size=batch_size, in_h=h, in_w=w, relu=True))
        h = (h + 2*p - k) // s + 1
        w = (w + 2*p - k) // s + 1

    final_c = layers_config[-1][1]
    if mlp_head:
        layers.append(VkGAPMLP(gpu, final_c, batch_size, hidden=hidden))
    else:
        layers.append(VkGAPLinear(gpu, final_c, batch_size))

    return VkModel(gpu, layers)


def build_beat_crnn(gpu: VkCompute, input_size: int = 216,
                    hidden_size: int = 48, n_classes: int = 3,
                    batch_size: int = 8, max_seq_len: int = 128,
                    seq_mode: bool = False,
                    proj_size: int = 32) -> VkModel:
    """Build the beat detection CRNN.

    Architecture: Linear(input→proj_size, ReLU) → GRU(proj_size→hidden) →
                  Linear(hidden→n_classes)

    Args:
        input_size: features per frame (e.g., 108 CQT + 108 diff = 216)
        hidden_size: GRU hidden units
        n_classes: output dimension. Original 3-class softmax used 3
            (non-beat, beat, downbeat). v1 single-head BCE used 1.
            v2 multi-head sigmoid BCE uses 3 (downbeat / any-beat /
            any-onset hierarchical sigmoids).
        proj_size: input projection dimension. v1 default 32 stayed
            tight; v2 bumps to 64 to give each of the 3 sigmoid heads
            more raw spectrum signal to develop distinct features from.
        batch_size: number of independent sequences
        max_seq_len: max BPTT sequence length
        seq_mode: if True, size the linear layers' output buffers for
            (B * max_seq_len) flat batch — needed when running training
            in sequence-mode (one dispatch per layer for the full T-step
            chunk via gru_seq_forward). GRU sizing is unchanged either
            way; it owns the sequence dimension internally.
    """
    linear_batch = batch_size * max_seq_len if seq_mode else batch_size

    layers: list[VkLayer] = [
        VkLinear(gpu, input_size, proj_size, linear_batch, relu=True),
        VkGRU(gpu, proj_size, hidden_size, batch_size, max_seq_len),
        VkLinear(gpu, hidden_size, n_classes, linear_batch, relu=False),
    ]

    return VkModel(gpu, layers)
