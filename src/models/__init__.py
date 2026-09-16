"""Model architectures for Siamese."""

from .attention_resnet34 import AttentionResNet34, CrossAttentionModule

__all__ = ["AttentionResNet34", "CrossAttentionModule"]
