"""Tests for VkConv2d per-input-channel gradient masking.

Used by the curriculum-training pipeline to freeze specific input
channels during fine-tuning (e.g. train the new channel while
holding the original channels' kernels stable). Mismatched shapes
or accidentally-skipped masking can silently produce wrong gradients
for hours of training — hence the unit tests.

GPU-required tests follow the same skip-if-no-GPU pattern as
test_vk_leaks.py — they'll be skipped on systems without a Vulkan
compute context.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from wallpaper_ml.vk_compute import VkCompute
    from wallpaper_ml import VkConv2d
    _gpu = VkCompute()
    HAS_GPU = True
except Exception:
    HAS_GPU = False
    _gpu = None
    VkConv2d = None


# ============================================================================
# Shape validation — runs without GPU (the check happens before any
# GPU interaction)
# ============================================================================

class TestShapeValidation:
    """The shape guard catches off-by-one errors that would silently
    produce wrong-channel masks."""

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_wrong_shape_raises(self):
        layer = VkConv2d(_gpu, in_channels=3, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        # in_channels=3 → mults must be shape (3,)
        with pytest.raises(ValueError, match=r'Expected shape \(3,\)'):
            layer.set_per_input_channel_lr_mult(np.array([1.0, 1.0]))

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_too_many_dims_raises(self):
        layer = VkConv2d(_gpu, in_channels=3, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        with pytest.raises(ValueError):
            layer.set_per_input_channel_lr_mult(
                np.array([[1.0, 1.0, 1.0]]))  # (1, 3) not (3,)

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_correct_shape_no_error(self):
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        # Should not raise
        layer.set_per_input_channel_lr_mult(np.array([1.0, 1.0, 1.0, 1.0]))


# ============================================================================
# All-ones short-circuit — the optimization that says "no masking
# needed" so the dispatch overhead is skipped entirely.
# ============================================================================

class TestAllOnesShortcircuit:
    """All-ones multipliers (= no masking) shouldn't allocate a buffer
    or set the _grad_mult_active flag. The pre_sgd_step call then
    silently no-ops, which matters for training perf when masking is
    rarely-or-never used."""

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_all_ones_leaves_inactive(self):
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        # Initial state: not active, no buffer
        assert layer._grad_mult_active is False
        assert layer._grad_mult_buf is None
        # All-ones should leave both unchanged
        layer.set_per_input_channel_lr_mult(np.ones(4, dtype=np.float32))
        assert layer._grad_mult_active is False
        assert layer._grad_mult_buf is None

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_all_ones_int_dtype_still_shortcircuits(self):
        # Caller might pass an int array; the function casts to float32
        # internally. np.allclose with int 1s should still short-circuit.
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        layer.set_per_input_channel_lr_mult(np.ones(4, dtype=np.int32))
        assert layer._grad_mult_active is False

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_near_ones_short_circuits(self):
        """allclose default tolerance — small numerical noise that
        wouldn't have any practical effect should also short-circuit
        rather than triggering the dispatch overhead."""
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        layer.set_per_input_channel_lr_mult(
            np.array([1.0, 1.0 + 1e-9, 1.0, 1.0 - 1e-9], dtype=np.float32))
        assert layer._grad_mult_active is False

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_revert_to_ones_disables(self):
        """Setting a mask then setting all-ones should disable masking
        (idempotent — caller can re-disable without checking state)."""
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        # Activate masking
        layer.set_per_input_channel_lr_mult(
            np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32))
        assert layer._grad_mult_active is True
        # Now revert to all-ones
        layer.set_per_input_channel_lr_mult(np.ones(4, dtype=np.float32))
        assert layer._grad_mult_active is False


# ============================================================================
# Active masking — non-uniform multipliers create the buffer and
# upload the values
# ============================================================================

class TestActiveMasking:
    """Non-uniform multipliers should allocate a GPU buffer and upload
    the mask values."""

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_non_uniform_activates(self):
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        layer.set_per_input_channel_lr_mult(
            np.array([0.0, 1.0, 1.0, 0.5], dtype=np.float32))
        assert layer._grad_mult_active is True
        assert layer._grad_mult_buf is not None

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_buffer_size_matches_in_channels(self):
        layer = VkConv2d(_gpu, in_channels=7, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        layer.set_per_input_channel_lr_mult(
            np.array([0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0], dtype=np.float32))
        # 7 floats × 4 bytes/float = 28 bytes
        assert layer._grad_mult_buf.size == 28

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_uploaded_values_roundtrip(self):
        """The mask we upload should be what we get back. Catches any
        byte-order / dtype-cast bug in the upload path."""
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        mults = np.array([0.1, 0.9, 0.0, 1.0], dtype=np.float32)
        layer.set_per_input_channel_lr_mult(mults)
        readback = _gpu.download(layer._grad_mult_buf, np.float32, 4)
        np.testing.assert_array_equal(mults, readback)

    @pytest.mark.skipif(not HAS_GPU, reason='No Vulkan GPU context')
    def test_remask_reuses_buffer(self):
        """Calling set_per_input_channel_lr_mult twice with different
        non-uniform values should reuse the buffer (not allocate a
        new one) — that buffer-allocation churn would mirror exactly
        the leak shape that motivated the test_vk_leaks suite."""
        layer = VkConv2d(_gpu, in_channels=4, out_channels=8,
                          kernel_size=3, stride=1, padding=1,
                          batch_size=1, in_h=32, in_w=32)
        layer.set_per_input_channel_lr_mult(
            np.array([0.0, 1.0, 1.0, 0.5], dtype=np.float32))
        buf_first = layer._grad_mult_buf
        layer.set_per_input_channel_lr_mult(
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
        buf_second = layer._grad_mult_buf
        assert buf_first is buf_second, 'buffer was reallocated (leak risk)'
