"""Vulkan compute layer primitives — PyTorch-style abstraction.

Each layer owns its buffers (weights, gradients, output, backward
scratch), dispatches shader pipelines for forward/backward, and
implements the VkLayer interface so a VkModel can chain them.
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .vk_compute import VkCompute, VkBuffer

SHADER_DIR = Path(__file__).parent / 'shaders' / 'cnn'
RNN_SHADER_DIR = Path(__file__).parent / 'shaders' / 'rnn'


# ---------------------------------------------------------------------------
# Base layer
# ---------------------------------------------------------------------------

class VkLayer(ABC):
    """Base class for Vulkan compute layers."""

    def __init__(self, gpu: VkCompute):
        self.gpu = gpu
        self._param_bufs: list[VkBuffer] = []    # float32 weight buffers
        self._grad_bufs: list[VkBuffer] = []      # uint32 gradient buffers (CAS)
        self._param_sizes: list[int] = []          # element counts per param

    @abstractmethod
    def forward(self, input_buf: VkBuffer, batch_size: int,
                shape: tuple[int, ...]) -> tuple[VkBuffer, tuple[int, ...]]:
        """Forward pass. Returns (output_buf, output_shape)."""
        ...

    @abstractmethod
    def backward(self, grad_output: VkBuffer, batch_size: int) -> VkBuffer:
        """Backward pass. Returns gradient w.r.t. input."""
        ...

    def zero_grad(self) -> None:
        for g in self._grad_bufs:
            self.gpu.zero_buffer(g)

    def parameters(self) -> list[tuple[VkBuffer, VkBuffer, int]]:
        """Returns [(param_buf, grad_buf, n_elements), ...]"""
        return list(zip(self._param_bufs, self._grad_bufs, self._param_sizes))

    def param_count(self) -> int:
        return sum(self._param_sizes)

    def load_weights(self, flat: np.ndarray, offset: int) -> int:
        """Load weights from flat array at offset. Returns new offset."""
        for buf, n in zip(self._param_bufs, self._param_sizes):
            self.gpu.upload(buf, flat[offset:offset + n].astype(np.float32))
            offset += n
        return offset

    def save_weights(self) -> np.ndarray:
        """Save weights to flat float32 array."""
        parts = []
        for buf, n in zip(self._param_bufs, self._param_sizes):
            parts.append(self.gpu.download(buf, np.float32, n))
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def init_weights(self, rng: np.random.Generator) -> None:
        """Initialize weights with Kaiming uniform. Override per layer."""
        pass


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class VkConv2d(VkLayer):
    """2D convolution + bias + optional ReLU."""

    def __init__(self, gpu: VkCompute, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int,
                 batch_size: int, in_h: int, in_w: int,
                 relu: bool = True):
        super().__init__(gpu)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.relu = relu

        self.in_h = in_h
        self.in_w = in_w
        self.out_h = (in_h + 2*padding - kernel_size) // stride + 1
        self.out_w = (in_w + 2*padding - kernel_size) // stride + 1
        self.batch_size = batch_size

        # Weight buffers
        n_kernel = out_channels * in_channels * kernel_size * kernel_size
        n_bias = out_channels
        self.kernel_buf = gpu.create_buffer(n_kernel * 4)
        self.bias_buf = gpu.create_buffer(n_bias * 4)
        self._param_bufs = [self.kernel_buf, self.bias_buf]
        self._param_sizes = [n_kernel, n_bias]

        # Gradient buffers (uint32 for CAS atomics)
        self.grad_buf = gpu.create_buffer((n_kernel + n_bias) * 4)
        self._grad_bufs = [self.grad_buf]

        # Output buffer
        self.output_buf = gpu.create_buffer(
            batch_size * out_channels * self.out_h * self.out_w * 4)

        # Backward buffer (reused across calls)
        self._d_input_buf = gpu.create_buffer(
            batch_size * in_channels * in_h * in_w * 4)

        # Saved for backward
        self._saved_input: VkBuffer | None = None
        self._saved_output: VkBuffer | None = None

    def forward(self, input_buf, batch_size, shape):
        B = batch_size
        self._saved_input = input_buf

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_forward.comp'),
            buffers=[input_buf, self.kernel_buf, self.bias_buf, self.output_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding,
                           1 if self.relu else 0)
        gx = (self.out_w + 7) // 8
        gy = (self.out_h + 7) // 8
        gz = B * self.out_channels
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        # Save output for ReLU mask in backward
        if self.relu:
            self._saved_output = self.output_buf

        return self.output_buf, (self.out_channels, self.out_h, self.out_w)

    def backward(self, grad_output, batch_size):
        B = batch_size

        # Apply ReLU mask to upstream gradient if needed
        if self.relu and self._saved_output is not None:
            d_relu_buf = grad_output  # shader handles mask via fwd_output
        else:
            d_relu_buf = grad_output

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_backward_input.comp'),
            buffers=[d_relu_buf, self.kernel_buf, self._d_input_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding, 0)
        gx = (self.in_w + 7) // 8
        gy = (self.in_h + 7) // 8
        gz = B * self.in_channels
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        # Backward weight gradient
        n_total = self._param_sizes[0] + self._param_sizes[1]
        fwd_buf = self._saved_output if self.relu else grad_output

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'conv2d_backward_weights.comp'),
            buffers=[grad_output, self._saved_input, self.grad_buf,
                     fwd_buf if self.relu else self.output_buf],
            push_constant_size=44,
        )
        push = struct.pack('11i', B, self.in_channels, self.out_channels,
                           self.in_h, self.in_w, self.out_h, self.out_w,
                           self.kernel_size, self.stride, self.padding,
                           1 if self.relu else 0)
        self.gpu.dispatch(pipeline, (n_total + 63) // 64, push_constants=push)
        pipeline.destroy()

        return self._d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def parameters(self):
        # Single combined grad buffer for kernel+bias
        n_k = self._param_sizes[0]
        n_b = self._param_sizes[1]
        return [(self.kernel_buf, self.grad_buf, n_k),
                (self.bias_buf, self.grad_buf, n_b)]

    def init_weights(self, rng):
        n_k = self._param_sizes[0]
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        std = float(np.sqrt(2.0 / fan_in))
        kernel = (rng.standard_normal(n_k) * std).astype(np.float32)
        # Zero bias init. We previously used +0.1 as a dead-ReLU escape
        # when domain channels had nonzero-mean inputs (sentinel-pixel
        # dominated). With v3 sentinel-aware standardization in place,
        # hit pixels are zero-mean / unit-std and sentinels snap to a
        # fixed out-of-distribution +5.0 — first-conv outputs no longer
        # land uniformly below zero. The +0.1 bias additionally caused a
        # systematic init bias: positive-weight filters fired hard on
        # sentinel pixels (passing ReLU due to +0.1 floor) while
        # negative-weight filters fired the same as without — so random
        # init had a positive coefficient on sentinel count. Sentinels
        # anti-correlate with quality (disliked genomes have ~7pt more
        # sentinels than liked), so the model started at 0.33 val_acc
        # instead of 0.50. Zero bias restores symmetry.
        bias = np.zeros(self.out_channels, dtype=np.float32)
        self.gpu.upload(self.kernel_buf, kernel)
        self.gpu.upload(self.bias_buf, bias)

    def save_weights(self):
        k = self.gpu.download(self.kernel_buf, np.float32, self._param_sizes[0])
        b = self.gpu.download(self.bias_buf, np.float32, self._param_sizes[1])
        return np.concatenate([k, b])

    def load_weights(self, flat, offset):
        n_k = self._param_sizes[0]
        n_b = self._param_sizes[1]
        self.gpu.upload(self.kernel_buf, flat[offset:offset+n_k].astype(np.float32))
        offset += n_k
        self.gpu.upload(self.bias_buf, flat[offset:offset+n_b].astype(np.float32))
        offset += n_b
        return offset


# ---------------------------------------------------------------------------
# GAP + Linear (fused)
# ---------------------------------------------------------------------------

class VkGAPLinear(VkLayer):
    """Global Average Pool + Linear(C, 1) → scalar score."""

    def __init__(self, gpu: VkCompute, channels: int, batch_size: int):
        super().__init__(gpu)
        self.channels = channels
        self.batch_size = batch_size

        n_w = channels + 1  # weights + bias
        self.weight_buf = gpu.create_buffer(n_w * 4)
        self._param_bufs = [self.weight_buf]
        self._param_sizes = [n_w]

        self.grad_buf = gpu.create_buffer(n_w * 4)
        self._grad_bufs = [self.grad_buf]

        self.pooled_buf = gpu.create_buffer(batch_size * channels * 4)
        self.output_buf = gpu.create_buffer(batch_size * 4)

        self._spatial_h = 0
        self._spatial_w = 0
        self._d_input_buf: VkBuffer | None = None

    def forward(self, input_buf, batch_size, shape):
        C, H, W = shape
        self._spatial_h = H
        self._spatial_w = W

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_linear_forward.comp'),
            buffers=[input_buf, self.weight_buf, self.pooled_buf, self.output_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, C, H, W)
        self.gpu.dispatch(pipeline, (batch_size + 31) // 32, push_constants=push)
        pipeline.destroy()

        return self.output_buf, (1,)

    def backward(self, grad_output, batch_size):
        C = self.channels
        H, W = self._spatial_h, self._spatial_w
        if self._d_input_buf is None:
            self._d_input_buf = self.gpu.create_buffer(batch_size * C * H * W * 4)

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_linear_backward.comp'),
            buffers=[grad_output, self.pooled_buf, self.weight_buf,
                     self._d_input_buf, self.grad_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, C, H, W)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = batch_size * C
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        return self._d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def init_weights(self, rng):
        C = self.channels
        std = float(np.sqrt(2.0 / C))
        w = (rng.standard_normal(C) * std).astype(np.float32)
        b = np.zeros(1, dtype=np.float32)
        self.gpu.upload(self.weight_buf, np.concatenate([w, b]))


# ---------------------------------------------------------------------------
# GAP + MLP (fused)
# ---------------------------------------------------------------------------

class VkGAPMLP(VkLayer):
    """Global Average Pool + Linear(C,H) + ReLU + Linear(H,1) → scalar."""

    def __init__(self, gpu: VkCompute, channels: int, batch_size: int,
                 hidden: int = 16):
        super().__init__(gpu)
        self.channels = channels
        self.hidden = hidden
        self.batch_size = batch_size

        n_w = channels * hidden + hidden + hidden + 1
        self.weight_buf = gpu.create_buffer(n_w * 4)
        self._param_bufs = [self.weight_buf]
        self._param_sizes = [n_w]

        self.grad_buf = gpu.create_buffer(n_w * 4)
        self._grad_bufs = [self.grad_buf]

        self.pooled_buf = gpu.create_buffer(batch_size * channels * 4)
        self.hidden_buf = gpu.create_buffer(batch_size * hidden * 4)
        self.output_buf = gpu.create_buffer(batch_size * 4)

        self._spatial_h = 0
        self._spatial_w = 0
        self._d_input_buf: VkBuffer | None = None

    def forward(self, input_buf, batch_size, shape):
        C, H, W = shape
        self._spatial_h = H
        self._spatial_w = W

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_mlp_forward.comp'),
            buffers=[input_buf, self.weight_buf, self.pooled_buf,
                     self.output_buf, self.hidden_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, C, H, W, self.hidden)
        self.gpu.dispatch(pipeline, (batch_size + 31) // 32, push_constants=push)
        pipeline.destroy()

        return self.output_buf, (1,)

    def backward(self, grad_output, batch_size):
        C = self.channels
        H, W = self._spatial_h, self._spatial_w
        if self._d_input_buf is None:
            self._d_input_buf = self.gpu.create_buffer(batch_size * C * H * W * 4)

        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'gap_mlp_backward.comp'),
            buffers=[grad_output, self.pooled_buf, self.weight_buf,
                     self._d_input_buf, self.grad_buf, self.hidden_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, C, H, W, self.hidden)
        gx = (W + 7) // 8
        gy = (H + 7) // 8
        gz = batch_size * C
        self.gpu.dispatch(pipeline, gx, gy, gz, push)
        pipeline.destroy()

        return self._d_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_buf)

    def init_weights(self, rng):
        C, H = self.channels, self.hidden
        std1 = float(np.sqrt(2.0 / C))
        w1 = (rng.standard_normal(C * H) * std1).astype(np.float32)
        b1 = np.zeros(H, dtype=np.float32)
        std2 = float(np.sqrt(2.0 / H))
        w2 = (rng.standard_normal(H) * std2).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        self.gpu.upload(self.weight_buf, np.concatenate([w1, b1, w2, b2]))


# ---------------------------------------------------------------------------
# Linear (RNN-style dense)
# ---------------------------------------------------------------------------

class VkLinear(VkLayer):
    """Dense linear layer: y = x @ W + b, optional ReLU.

    Input:  [B, in_features]  (shape = (in_features,))
    Output: [B, out_features] (shape = (out_features,))
    """

    def __init__(self, gpu: VkCompute, in_features: int, out_features: int,
                 batch_size: int, relu: bool = False):
        super().__init__(gpu)
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.relu = relu

        # Weight: [in_features, out_features], Bias: [out_features]
        n_w = in_features * out_features
        n_b = out_features
        self.weight_buf = gpu.create_buffer(n_w * 4)
        self.bias_buf = gpu.create_buffer(n_b * 4)
        self._param_bufs = [self.weight_buf, self.bias_buf]
        self._param_sizes = [n_w, n_b]

        # Gradient buffers (uint32 CAS)
        self.grad_weight_buf = gpu.create_buffer(n_w * 4)
        self.grad_bias_buf = gpu.create_buffer(n_b * 4)
        self._grad_bufs = [self.grad_weight_buf, self.grad_bias_buf]

        # Output buffer
        self.output_buf = gpu.create_buffer(batch_size * out_features * 4)

        # Saved for backward
        self._saved_input: VkBuffer | None = None

    def forward(self, input_buf, batch_size, shape, input_offset_floats: int = 0):
        """input_offset_floats: float-element offset into input_buf. Lets the
        caller pre-upload a (B, T, in_features) sequence to a larger buffer
        and slice per-timestep via this offset instead of re-uploading just
        that slice each call. Default 0 keeps existing single-frame callers
        unchanged."""
        self._saved_input = input_buf
        self._saved_input_offset = input_offset_floats

        pipeline = self.gpu.create_pipeline(
            str(RNN_SHADER_DIR / 'linear_forward.comp'),
            buffers=[input_buf, self.weight_buf, self.bias_buf, self.output_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, self.in_features,
                           self.out_features, 1 if self.relu else 0,
                           input_offset_floats)
        total = batch_size * self.out_features
        self.gpu.dispatch(pipeline, (total + 63) // 64, push_constants=push)
        pipeline.destroy()

        if self.relu:
            self._saved_output = self.output_buf

        return self.output_buf, (self.out_features,)

    def backward(self, grad_output, batch_size):
        B = batch_size
        I, O = self.in_features, self.out_features

        # Need saved forward output for ReLU mask
        fwd_out_buf = self._saved_output if self.relu else self.output_buf

        # Allocate grad_input buffer
        if not hasattr(self, '_grad_input_buf') or self._grad_input_buf is None:
            self._grad_input_buf = self.gpu.create_buffer(B * I * 4)

        pipeline = self.gpu.create_pipeline(
            str(RNN_SHADER_DIR / 'linear_backward.comp'),
            buffers=[grad_output, self._saved_input, self.weight_buf,
                     self._grad_input_buf, self.grad_weight_buf,
                     self.grad_bias_buf, fwd_out_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', B, I, O, 1 if self.relu else 0)
        total = B * I
        self.gpu.dispatch(pipeline, (total + 63) // 64, push_constants=push)
        pipeline.destroy()

        return self._grad_input_buf

    def zero_grad(self):
        self.gpu.zero_buffer(self.grad_weight_buf)
        self.gpu.zero_buffer(self.grad_bias_buf)

    def init_weights(self, rng):
        std = float(np.sqrt(2.0 / self.in_features))
        w = (rng.standard_normal(self.in_features * self.out_features) * std).astype(np.float32)
        b = np.zeros(self.out_features, dtype=np.float32)
        self.gpu.upload(self.weight_buf, w)
        self.gpu.upload(self.bias_buf, b)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class VkGRU(VkLayer):
    """GRU recurrent layer — single-frame or sequence forward.

    Per frame: takes [B, input_size], updates hidden [B, hidden_size],
    outputs [B, hidden_size].

    For BPTT training, caches activations per timestep.
    """

    def __init__(self, gpu: VkCompute, input_size: int, hidden_size: int,
                 batch_size: int, max_seq_len: int = 128):
        super().__init__(gpu)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        H = hidden_size
        I = input_size

        # W: [3, input_size, hidden_size] — gates z, r, h
        n_W = 3 * I * H
        # U: [3, hidden_size, hidden_size]
        n_U = 3 * H * H
        # bias: [6, hidden_size] — split input/hidden biases (PyTorch compat)
        n_bias = 6 * H

        self.W_buf = gpu.create_buffer(n_W * 4)
        self.U_buf = gpu.create_buffer(n_U * 4)
        self.bias_buf = gpu.create_buffer(n_bias * 4)
        self._param_bufs = [self.W_buf, self.U_buf, self.bias_buf]
        self._param_sizes = [n_W, n_U, n_bias]

        # Gradient buffers
        self.grad_W_buf = gpu.create_buffer(n_W * 4)
        self.grad_U_buf = gpu.create_buffer(n_U * 4)
        self.grad_bias_buf = gpu.create_buffer(n_bias * 4)
        self._grad_bufs = [self.grad_W_buf, self.grad_U_buf, self.grad_bias_buf]

        # Hidden state (persists across frames)
        self.hidden_buf = gpu.create_buffer(batch_size * H * 4)

        # Output buffer (copy of hidden after forward)
        self.output_buf = gpu.create_buffer(batch_size * H * 4)

        # Activation cache for BPTT: [max_seq_len, B, 4*H]
        self.cache_buf = gpu.create_buffer(max_seq_len * batch_size * 4 * H * 4)
        self._seq_pos = 0  # current position in cache

        # Pre-allocate grad_input_buf at the worst-case size — needed for
        # seq-mode BPTT where each iteration writes its own timestep's
        # input gradient to a (T, B, I) buffer. Single-step backward only
        # uses the first (B, I) of it.
        self._grad_input_buf = gpu.create_buffer(max_seq_len * batch_size * I * 4)
        self._dh_buf = gpu.create_buffer(batch_size * H * 4)

        # Flag set by forward_sequence so backward knows to advance offsets
        # rather than indexing into a list of per-step input buffers.
        self._seq_mode = False

    def reset_hidden(self) -> None:
        """Zero the hidden state (call on new song, etc.)."""
        self.gpu.zero_buffer(self.hidden_buf)
        self._seq_pos = 0
        self._saved_inputs: list[VkBuffer] = []
        self._seq_mode = False

    def forward_sequence(self, input_seq_buf, output_seq_buf,
                         batch_size: int, T: int) -> VkBuffer:
        """Run forward for T timesteps in a single dispatch via gru_seq_forward.

        input_seq_buf:  (T, B, input_size) — caller pre-uploaded
        output_seq_buf: (T, B, hidden_size) — written, caller-allocated
        batch_size: number of independent sequences (rows of B)
        T: number of timesteps to process this call

        Replaces T calls of single-step forward() with one dispatch.
        Cache layout is identical to single-step so the existing
        gru_backward shader works on it directly (with seq-mode offsets).

        Hidden state in self.hidden_buf is used as initial state and
        updated to the final state of the sequence — supports chained
        sub-batches (call with t_start=0, t_count=T1, then t_start=T1,
        t_count=T2, ...).
        """
        pipeline = self.gpu.create_pipeline(
            str(RNN_SHADER_DIR / 'gru_seq_forward.comp'),
            buffers=[input_seq_buf, self.hidden_buf, self.W_buf, self.U_buf,
                     self.bias_buf, output_seq_buf, self.cache_buf],
            push_constant_size=20,
        )
        push = struct.pack('5i', batch_size, self.input_size,
                           self.hidden_size, 0, T)
        self.gpu.dispatch(pipeline, batch_size, push_constants=push)
        pipeline.destroy()

        # Tell backward to use seq-mode offsetting: one big input buffer,
        # one big grad_input buffer, T BPTT iterations with advancing offsets.
        self._saved_inputs = [input_seq_buf]
        self._seq_mode = True
        self._seq_pos = T
        return output_seq_buf

    def backward_sequence(self, upstream_seq_buf: 'VkBuffer',
                          batch_size: int, T: int) -> 'VkBuffer':
        """Single-dispatch BPTT over T timesteps via gru_seq_backward.comp.

        upstream_seq_buf: (T, B, hidden) gradient on GRU output for every
            timestep (from the layer above — typically linear_out's
            _grad_input_buf after its B*T flat-batch backward).
        batch_size: B (number of sequences)
        T: matches the forward_sequence T

        Replaces T calls of per-timestep gru_backward + the per-step
        upload of upstream dh. All BPTT iterations run inside one
        workgroup with the recurrent dh in shared memory. True
        per-timestep upstream — every timestep's gradient gets added
        to the running recurrent grad, not just the last (which was
        the BPTT approximation that limited the previous backward path).

        Returns self._grad_input_buf — (T, B, input_size) per-timestep
        gradient on input. Weight gradients accumulate atomically into
        grad_W / grad_U / grad_bias just like the per-step shader.
        """
        if not self._seq_mode or self._saved_inputs is None or not self._saved_inputs:
            raise RuntimeError('backward_sequence requires forward_sequence first')

        pipeline = self.gpu.create_pipeline(
            str(RNN_SHADER_DIR / 'gru_seq_backward.comp'),
            buffers=[self._saved_inputs[0], upstream_seq_buf,
                     self._grad_input_buf, self.cache_buf,
                     self.W_buf, self.U_buf, self.bias_buf,
                     self.grad_W_buf, self.grad_U_buf, self.grad_bias_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, self.input_size,
                           self.hidden_size, T)
        self.gpu.dispatch(pipeline, batch_size, push_constants=push)
        pipeline.destroy()
        return self._grad_input_buf

    def forward(self, input_buf, batch_size, shape):
        """Single-frame GRU forward. Updates hidden state in-place."""
        H = self.hidden_size

        # Save input reference for backward
        if not hasattr(self, '_saved_inputs'):
            self._saved_inputs = []
        self._saved_inputs.append(input_buf)

        # Cache offset for this timestep
        cache_offset = self._seq_pos * batch_size * 4 * H

        pipeline = self.gpu.create_pipeline(
            str(RNN_SHADER_DIR / 'gru_forward.comp'),
            buffers=[input_buf, self.hidden_buf, self.W_buf, self.U_buf,
                     self.bias_buf, self.output_buf, self.cache_buf],
            push_constant_size=16,
        )
        push = struct.pack('4i', batch_size, self.input_size,
                           self.hidden_size, cache_offset)
        # One workgroup per batch element, local_size_x covers hidden units
        self.gpu.dispatch(pipeline, batch_size, push_constants=push)
        pipeline.destroy()

        self._seq_pos += 1

        return self.output_buf, (self.hidden_size,)


    def backward(self, grad_output, batch_size):
        """BPTT backward over cached timesteps.

        grad_output: [B, H] gradient from the layer above (final timestep)
        OR for sequence training, this should be called per-timestep from T-1 to 0.

        For the VkModel backward() API, we handle the full BPTT here
        using the cached activations from forward_sequence or sequential forward calls.
        """
        B = batch_size
        H = self.hidden_size
        I = self.input_size
        T = self._seq_pos  # number of timesteps we forwarded

        if T == 0:
            # No timesteps cached, nothing to backprop
            if not hasattr(self, '_grad_input_buf') or self._grad_input_buf is None:
                self._grad_input_buf = self.gpu.create_buffer(B * I * 4)
            self.gpu.zero_buffer(self._grad_input_buf)
            return self._grad_input_buf

        # dh_buf holds the running gradient w.r.t. hidden state
        if not hasattr(self, '_dh_buf') or self._dh_buf is None:
            self._dh_buf = self.gpu.create_buffer(B * H * 4)
        dh_data = self.gpu.download(grad_output, np.float32, B * H)
        self.gpu.upload(self._dh_buf, dh_data)

        # dx buffer for each timestep (we only need to output the final one
        # for VkModel.backward(), but for sequence training we'd accumulate)
        if not hasattr(self, '_grad_input_buf') or self._grad_input_buf is None:
            self._grad_input_buf = self.gpu.create_buffer(B * I * 4)

        # BPTT: iterate backwards through cached timesteps
        #
        # In single-step mode (one input_buf per timestep, output to one
        # (B, I) grad buffer): input_offset/grad_x_offset stay at 0 every
        # iter, _saved_inputs[t] points to a different small buffer each t.
        #
        # In seq mode (single (T, B, I) input buffer, single (T, B, I)
        # grad buffer): _seq_mode is set, _saved_inputs[t] is the same big
        # buffer every t, and offsets advance by B*I per timestep.
        seq_mode = getattr(self, '_seq_mode', False)

        for t in range(T - 1, -1, -1):
            cache_offset = t * B * 4 * H

            if seq_mode:
                # Same buffer every iteration, offset advances
                input_buf = self._saved_inputs[0]
                input_offset = t * B * I
                grad_x_offset = t * B * I
            else:
                # Distinct per-timestep buffer; no offset
                input_buf = (self._saved_inputs[t]
                             if hasattr(self, '_saved_inputs') and self._saved_inputs
                             else self._saved_input)
                input_offset = 0
                grad_x_offset = 0

            pipeline = self.gpu.create_pipeline(
                str(RNN_SHADER_DIR / 'gru_backward.comp'),
                buffers=[input_buf, self._dh_buf, self._grad_input_buf,
                         self.cache_buf, self.W_buf, self.U_buf,
                         self.bias_buf, self.grad_W_buf,
                         self.grad_U_buf, self.grad_bias_buf],
                push_constant_size=24,
            )
            push = struct.pack('6i', B, I, H, cache_offset,
                               input_offset, grad_x_offset)
            self.gpu.dispatch(pipeline, B, push_constants=push)
            pipeline.destroy()

            # dh_buf now contains dh_prev for this timestep
            # (written in-place by the shader)

        return self._grad_input_buf

    def init_weights(self, rng):
        """Initialize GRU weights — Xavier uniform for gates."""
        H = self.hidden_size
        I = self.input_size

        # W: [3, I, H] — Xavier uniform based on fan_in=I
        std_w = float(np.sqrt(1.0 / I))
        W = (rng.standard_normal(3 * I * H) * std_w).astype(np.float32)

        # U: [3, H, H] — orthogonal-ish init (Xavier with fan_in=H)
        std_u = float(np.sqrt(1.0 / H))
        U_arr = (rng.standard_normal(3 * H * H) * std_u).astype(np.float32)

        # Bias: [6, H] — zeros, except bias for reset gate slightly positive
        # to encourage remembering at init
        bias = np.zeros(6 * H, dtype=np.float32)

        self.gpu.upload(self.W_buf, W)
        self.gpu.upload(self.U_buf, U_arr)
        self.gpu.upload(self.bias_buf, bias)
