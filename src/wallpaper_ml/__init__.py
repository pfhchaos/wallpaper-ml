"""Vulkan compute ML framework — PyTorch-style layer abstraction.

Wraps the raw VkCompute shader dispatch into composable layers with
automatic buffer management, weight serialization, and gradient tracking.

Usage:
    from wallpaper_ml import VkCompute, build_cnn_scorer
    gpu = VkCompute()
    model = build_cnn_scorer(gpu, layers_config, batch_size=8)
    model.load_weights(np.load('weights.npy'))
    scores = model.forward(input_buf)
    model.backward(grad_buf)
    model.sgd_step(lr=0.003)
    np.save('weights.npy', model.save_weights())
"""

from .vk_compute import VkCompute, VkBuffer
from .layers import (
    VkLayer,
    VkConv2d,
    VkGAPLinear,
    VkGAPMLP,
    VkLinear,
    VkGRU,
)
from .losses import bce_loss_dispatch, cross_entropy_dispatch
from .models import VkModel, build_cnn_scorer, build_beat_crnn

__all__ = [
    'VkCompute',
    'VkBuffer',
    'VkLayer',
    'VkConv2d',
    'VkGAPLinear',
    'VkGAPMLP',
    'VkLinear',
    'VkGRU',
    'VkModel',
    'bce_loss_dispatch',
    'cross_entropy_dispatch',
    'build_cnn_scorer',
    'build_beat_crnn',
]
