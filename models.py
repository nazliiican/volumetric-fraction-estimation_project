"""PyTorch model for the transfer-learning composition experiment.

A pretrained EfficientNet-B0 backbone (frozen) produces a fixed image
embedding; a small softmax head turns that embedding into a valid composition
vector. Keeping the backbone frozen is deliberate: with only 805 images from
161 mixtures there is not enough data to fine-tune 5M backbone weights without
overfitting, so to train just the tiny head.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_efficientnet_backbone(pretrained=True, freeze=True):
    """Return an EfficientNet-B0 feature extractor and its embedding dimension."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        net = models.efficientnet_b0(weights=weights)
    except Exception as exc:  
        print(f"[models] could not load pretrained weights ({exc}); using random init.")
        net = models.efficientnet_b0(weights=None)

    embedding_dim = net.classifier[1].in_features  
    net.classifier = nn.Sequential(nn.Identity())                

    if freeze:
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

    return net, embedding_dim


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CompositionCNN(nn.Module):
    """EfficientNet-B0 backbone + dropout + linear head + softmax.

    The softmax output is, by construction, a valid composition vector: every
    entry is non-negative and each row sums to 1.
    """
    def __init__(self, backbone, embedding_dim, n_materials, dropout=0.3,
                 freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, n_materials)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.backbone(x)
        else:
            embedding = self.backbone(x)
        logits = self.head(self.dropout(embedding))
        return torch.softmax(logits, dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class SparseMultitaskCNN(nn.Module):
    """Shared frozen backbone with separate presence and composition heads.

    Both heads read the same embedding. The composition head produces a softmax
    distribution over materials; the presence head produces a per-material
    probability.Formula:
        fused = softmax(composition_logits) * sigmoid(presence_logits)
        fused = fused / (sum(fused) + eps)"""
    def __init__(self, backbone, embedding_dim, n_materials, dropout=0.3,
                 freeze_backbone=True, eps=1e-8):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        self.presence_head = nn.Linear(embedding_dim, n_materials)
        self.composition_head = nn.Linear(embedding_dim, n_materials)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.backbone(x)
        else:
            embedding = self.backbone(x)
        embedding = self.dropout(embedding)

        presence_logits = self.presence_head(embedding)
        presence_prob = torch.sigmoid(presence_logits)
        composition = torch.softmax(self.composition_head(embedding), dim=1)

        fused = composition * presence_prob
        fused = fused / (fused.sum(dim=1, keepdim=True) + self.eps)

        return {
            "presence_logits": presence_logits,
            "presence_prob": presence_prob,
            "composition": composition,
            "fused_composition": fused,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
