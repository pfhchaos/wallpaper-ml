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


class MultiDepthBeatRNN(VkModel):
    """Stacked-GRU beat RNN with heads at multiple depths.

    Forward shape:

        input (B*T, F) → input_linear (B*T, P) → gru[0] (T,B,H) ─┬─→ head[0] (B*T,1)
                                                                ↓
                                                            gru[1] (T,B,H) ─┬─→ head[1] (B*T,1)
                                                                            ↓
                                                                        gru[k-1] (T,B,H) ──→ head[k-1] (B*T,1)

    Each head reads from its own GRU layer's output so different heads
    can learn at different temporal-context depths (the motivation:
    onset cues are local, downbeat cues require longer memory). All
    heads are independent sigmoids — bce_loss_dispatch once per head.

    Storage order in self.layers (matches save_weights concat layout):
        [input_linear, gru_0, gru_1, ..., gru_{k-1}, head_0, ..., head_{k-1}]
    This is also the legacy lookup-table key — the runtime BeatRNNDetector
    loader sums per-layer param counts in this order. Keep it stable.

    head/gru index convention: index 0 = shallowest layer, index k-1 = deepest.
    The convention for which task each head handles is set by the trainer
    (default: onset=0, beat=1, downbeat=2 for k=3), not baked into this class.
    """

    def __init__(self, gpu: VkCompute,
                 input_linear: 'VkLinear',
                 grus: list['VkGRU'],
                 heads: list['VkLinear'],
                 batch_size: int, max_seq_len: int):
        if len(grus) != len(heads):
            raise ValueError(f'gru/head count mismatch: {len(grus)} vs {len(heads)}')
        if len(grus) < 2:
            raise ValueError(f'multi-depth requires ≥2 GRU layers, got {len(grus)}')
        layers: list[VkLayer] = [input_linear, *grus, *heads]
        super().__init__(gpu, layers)
        self.input_linear = input_linear
        self.grus = grus
        self.heads = heads
        self.n_gru_layers = len(grus)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # Per-layer GRU sequence-output buffer. forward_sequence writes
        # (T, B, hidden) into these; the next GRU consumes the previous
        # buffer as input. Pre-allocated at worst-case (max_seq_len * B).
        # Owning these here means the trainer doesn't need to know about
        # stack depth.
        self._gru_seq_outs: list[VkBuffer] = [
            gpu.create_buffer(max_seq_len * batch_size * g.hidden_size * 4)
            for g in grus
        ]

        # Cached add-inplace pipeline. Reused across the per-batch
        # backward — same shader, same buffer-count, only the push
        # constant (element count) varies.
        self._add_inplace_pipeline_cache: dict[int, object] = {}

    def reset_hidden(self) -> None:
        """Zero all GRU hidden states. Call between songs / training chunks."""
        for g in self.grus:
            g.reset_hidden()

    def _add_inplace(self, dst: VkBuffer, src: VkBuffer, count: int) -> None:
        """dst[i] += src[i] for i in [0, count). Used to sum gradient
        flows where two paths meet at the same hidden activation."""
        pipeline = self.gpu.create_pipeline(
            str(SHADER_DIR / 'add_inplace.comp'),
            buffers=[dst, src],
            push_constant_size=4,
        )
        push = struct.pack('I', count)
        self.gpu.dispatch(pipeline, (count + 63) // 64, push_constants=push)
        pipeline.destroy()

    def forward_multidepth(self, input_seq_buf: VkBuffer, B: int, T: int
                           ) -> list[VkBuffer]:
        """Run the stack and return one head-logit buffer per head.

        input_seq_buf: pre-uploaded (T, B, input_size) features.
        Returns: [head_0_logits, ...] — each shape (B*T, 1) inside the
                 caller-readable head linear's output_buf.

        The output buffers belong to the respective head VkLinears
        (head.output_buf) — callers should treat them as read-only and
        not free them; they get reused next forward call.
        """
        BT = B * T
        # 1. Input projection: (T, B, F) → (B*T, P)
        self.input_linear.forward(input_seq_buf, BT,
                                  (self.input_linear.in_features,))

        # 2. Stacked GRUs. Each GRU consumes the previous output buffer
        #    (or the input projection for gru[0]).
        prev_buf = self.input_linear.output_buf
        for gru, out_buf in zip(self.grus, self._gru_seq_outs):
            gru.forward_sequence(prev_buf, out_buf, B, T)
            prev_buf = out_buf

        # 3. Heads. Each head reads from its own GRU's output buffer.
        head_outs: list[VkBuffer] = []
        for head, gru_out in zip(self.heads, self._gru_seq_outs):
            out, _ = head.forward(gru_out, BT, (head.in_features,))
            head_outs.append(out)
        return head_outs

    def backward_multidepth(self, head_grad_bufs: list[VkBuffer],
                            B: int, T: int) -> None:
        """Backprop the per-head gradients through the stack.

        head_grad_bufs: [grad_logits_0, ...] — one buffer per head,
            each shape (B*T, 1), already populated by per-head bce_loss
            dispatches.

        Gradient routing at each intermediate hidden state h_k:
            dL/dh_k = head_k.backward(grad_logits_k)
                    + gru_{k+1}.backward_sequence(dL/dh_{k+1}).output

        Implementation: head_k's backward writes into head_k._grad_input_buf;
        gru_{k+1}'s backward writes into gru_{k+1}._grad_input_buf. We
        sum them in-place into head_k._grad_input_buf (the "accumulator"
        side), then feed that to gru_k.backward_sequence.

        Deepest layer has no upstream GRU — its dL/dh is just head's grad.
        Shallowest layer's gru_input is the input_linear output, no second
        path to sum.
        """
        BT = B * T
        H = self.grus[0].hidden_size  # all GRU layers share hidden_size today

        # Per-head backward: each head_k.backward writes its dL/dh_k into
        # head_k._grad_input_buf — shape (B*T, H) flat = (T, B, H) ravelled.
        for head, head_grad in zip(self.heads, head_grad_bufs):
            head.backward(head_grad, BT)

        # Walk the stack from deepest to shallowest.
        # At depth k (counting from 0=shallowest), the upstream into
        # gru_k.backward_sequence is dL/dh_k, which lives in
        # head_k._grad_input_buf (after summing in the contribution from
        # gru_{k+1} if k < n-1).
        for k in range(self.n_gru_layers - 1, -1, -1):
            # head_k._grad_input_buf already holds head_k's contribution.
            # If there's a GRU above, sum in its grad-on-input.
            if k < self.n_gru_layers - 1:
                # gru_{k+1}._grad_input_buf was written by the previous
                # iteration's backward_sequence — shape (T, B, H).
                upstream_gru = self.grus[k + 1]._grad_input_buf
                self._add_inplace(
                    self.heads[k]._grad_input_buf,
                    upstream_gru,
                    T * B * H,
                )
            self.grus[k].backward_sequence(
                self.heads[k]._grad_input_buf, B, T)

        # Final: input_linear's upstream is gru_0's grad-on-input.
        self.input_linear.backward(self.grus[0]._grad_input_buf, BT)

    # ---- Disable the linear sequential forward/backward inherited from
    # VkModel: the routing here is non-sequential, callers must use the
    # _multidepth variants instead.

    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError(
            'MultiDepthBeatRNN: use forward_multidepth() — the inherited '
            'sequential VkModel.forward() does not match this graph.')

    def backward(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError(
            'MultiDepthBeatRNN: use backward_multidepth() — the inherited '
            'sequential VkModel.backward() does not match this graph.')


def build_beat_crnn_multidepth(
        gpu: VkCompute, input_size: int = 216, hidden_size: int = 48,
        proj_size: int = 64, n_gru_layers: int = 3, n_heads: int = 3,
        batch_size: int = 8, max_seq_len: int = 128) -> MultiDepthBeatRNN:
    """Build a multi-depth stacked-GRU beat RNN.

    Architecture:
        Linear(input → proj, ReLU)
          → GRU_0(proj → hidden)  ──→ head_0 (hidden → 1)
          → GRU_1(hidden → hidden) ──→ head_1 (hidden → 1)
          → ... (n_gru_layers total)
          → GRU_{k-1}(hidden → hidden) ──→ head_{k-1} (hidden → 1)

    The seq-mode training pattern is the same as build_beat_crnn — one
    forward_sequence dispatch per GRU layer, one BCE loss per head, and
    one backward_sequence per GRU layer (gradients summed at each hidden
    state where head + upstream-GRU meet).

    Args:
        input_size: per-frame feature width (216 = 108 CQT + 108 diff)
        hidden_size: GRU hidden units. Same for every layer in the stack;
            mixed-width would need separate sizing per layer (not today).
        proj_size: input projection dim — only the first GRU sees it;
            subsequent GRUs consume hidden_size outputs.
        n_gru_layers: stack depth. With n_heads == n_gru_layers (default
            and intended use), each layer feeds its own head.
        n_heads: must equal n_gru_layers — the multi-depth design pins
            one head per layer. Kept as an explicit arg so the call site
            documents the intent.
        batch_size, max_seq_len: pre-allocation sizing for the GRU and
            linear output buffers.
    """
    if n_heads != n_gru_layers:
        raise ValueError(
            f'n_heads ({n_heads}) must equal n_gru_layers ({n_gru_layers}) — '
            'multi-depth pins one head per layer')

    BT = batch_size * max_seq_len  # flat-batch dim for the linear layers

    input_linear = VkLinear(gpu, input_size, proj_size, BT, relu=True)

    grus: list[VkLayer] = []
    for k in range(n_gru_layers):
        gru_in = proj_size if k == 0 else hidden_size
        grus.append(VkGRU(gpu, gru_in, hidden_size, batch_size, max_seq_len))

    heads: list[VkLayer] = [
        VkLinear(gpu, hidden_size, 1, BT, relu=False)
        for _ in range(n_heads)
    ]

    return MultiDepthBeatRNN(gpu, input_linear, grus, heads,
                             batch_size=batch_size, max_seq_len=max_seq_len)


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
