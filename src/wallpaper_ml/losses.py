"""Loss dispatch helpers — bce_loss, cross_entropy.

These are not VkLayer subclasses because losses don't have weights or
participate in the model's forward/backward chain. They write per-batch
loss values and gradients into pre-allocated accumulators that the
training loop owns.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .vk_compute import VkCompute, VkBuffer

RNN_SHADER_DIR = Path(__file__).parent / 'shaders' / 'rnn'


def bce_loss_dispatch(gpu: VkCompute, logits_buf: VkBuffer,
                      target_buf: VkBuffer, grad_acc_buf: VkBuffer,
                      loss_acc_buf: VkBuffer, batch_size: int,
                      target_offset_floats: int, t_inv: float,
                      pos_weight: float = 1.0) -> None:
    """Dispatch the bce_loss shader for one timestep.

    Single-channel regression: each batch element has one logit and one
    soft target in [0, 1]. Accumulates per-element loss (size B) and
    gradient (size B) into the pre-allocated accumulators, weighted by
    t_inv so after T dispatches the accumulators hold mean-over-
    timesteps values. Caller zeroes both at the start of the batch.

    pos_weight: PyTorch-style multiplier on positive (target=1)
    examples. 1.0 (default) = standard BCE. Use n_neg/n_pos to fully
    equalize positive/negative loss contribution within this head;
    sqrt(n_neg/n_pos) for a gentler rebalancing that doesn't risk
    flipping the bias the other way. Set per-head when the multidepth
    trainer's heads have different positive rates (downbeat is ~1.5%
    vs onset ~12% in the beat corpus).

    Used by the continuous-activation reformulation of the beat-RNN
    (single output channel = beat probability) — sidesteps the
    multi-class imbalance that broke the 3-class softmax training
    even with sqrt-inv-freq weights + focal loss.
    """
    pipeline = gpu.create_pipeline(
        str(RNN_SHADER_DIR / 'bce_loss.comp'),
        buffers=[logits_buf, target_buf, grad_acc_buf, loss_acc_buf],
        push_constant_size=16,
    )
    push = struct.pack('2iff', batch_size, target_offset_floats, t_inv,
                       float(pos_weight))
    n_workgroups = (batch_size + 31) // 32
    gpu.dispatch(pipeline, n_workgroups, push_constants=push)


def bce_loss_multich_dispatch(gpu: VkCompute, logits_buf: VkBuffer,
                               target_buf: VkBuffer, grad_acc_buf: VkBuffer,
                               loss_acc_buf: VkBuffer, batch_size: int,
                               n_classes: int, target_offset_floats: int,
                               t_inv: float) -> None:
    """Dispatch the multi-channel BCE shader — N independent sigmoid heads
    per batch element, per-channel BCE against soft targets.

    Buffer layouts:
      logits_buf:    (B * n_classes,) — logits[b * n_classes + c]
      target_buf:    sized for caller's full (B, T, C) array; the shader
                     reads targets[target_offset_floats + b*n_classes + c]
      grad_acc_buf:  (B * n_classes,) — accumulated gradients per channel
      loss_acc_buf:  (B,) — per-element loss SUMMED over channels

    Used by the 3-head beat-RNN (downbeat / any-beat / any-onset). Each
    head is an independent binary classifier — no softmax constraint, no
    class-imbalance pathology that killed the original 3-class softmax.

    Per-channel loss breakout is deferred to download time (caller can
    re-run the shader with single channels, or read individual gradients
    pre-accumulation if they need per-class F1 during training rather
    than at validation only).
    """
    pipeline = gpu.create_pipeline(
        str(RNN_SHADER_DIR / 'bce_loss_multich.comp'),
        buffers=[logits_buf, target_buf, grad_acc_buf, loss_acc_buf],
        push_constant_size=16,
    )
    push = struct.pack('3if', batch_size, n_classes, target_offset_floats, t_inv)
    n_workgroups = (batch_size + 31) // 32
    gpu.dispatch(pipeline, n_workgroups, push_constants=push)


def cross_entropy_dispatch(gpu: VkCompute, logits_buf: VkBuffer,
                            target_buf: VkBuffer, grad_acc_buf: VkBuffer,
                            loss_acc_buf: VkBuffer,
                            class_weights_buf: VkBuffer,
                            batch_size: int, n_classes: int,
                            target_offset_floats: int,
                            t_inv: float, focal_gamma: float = 0.0) -> None:
    """Dispatch the cross_entropy shader for one timestep.

    Accumulates loss into loss_acc_buf (size B floats) and gradient into
    grad_acc_buf (size B*n_classes floats), both weighted by t_inv so
    after T dispatches they hold mean-over-timesteps values. Caller is
    responsible for zeroing both accumulators at the start of the batch.

    class_weights_buf: (n_classes,) float32 — per-class multipliers
    applied to both loss and gradient. Use (1, 1, 1) for unweighted CE.
    For class-imbalanced data like beat detection, sqrt-inverse-frequency
    weights (e.g., ~6.8, ~3.9, ~1.0 for the 2%/7%/91% distribution)
    prevent majority-class collapse without the gradient instability of
    pure inverse-frequency weights.

    focal_gamma: 0 disables focal loss (standard weighted CE). 2.0
    (Lin 2017 default) penalizes confident-correct predictions —
    targets the "model is too sure of the easy majority class"
    failure mode directly. Pairs naturally with class weights: weights
    fix under-representation in loss; focal fixes over-confidence.

    target_offset_floats: offset into target_buf (in float elements) for
    the current timestep — lets the caller share one (B, T, C) labels
    buffer across all T dispatches instead of uploading per-timestep
    slices.
    """
    pipeline = gpu.create_pipeline(
        str(RNN_SHADER_DIR / 'cross_entropy.comp'),
        buffers=[logits_buf, target_buf, grad_acc_buf, loss_acc_buf,
                 class_weights_buf],
        push_constant_size=20,
    )
    push = struct.pack('3iff', batch_size, n_classes,
                       target_offset_floats, t_inv, focal_gamma)
    n_workgroups = (batch_size + 31) // 32  # matches local_size_x = 32
    gpu.dispatch(pipeline, n_workgroups, push_constants=push)
