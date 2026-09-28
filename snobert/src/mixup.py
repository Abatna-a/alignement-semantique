"""Augmented Mixup Procedure for Privacy-Preserving Collaborative Training,
adaptée aux embeddings de tokens SnoBERT.

Logique source : utils.py de l'implémentation de référence
  mix_feats = lam * f_i + (1-lam) * (f_j + noise)
  ||noise||_2 = radius  (uniforme sur la boule L2)
  radius = mf(tau, alpha, c) * r
  soft labels = lam * y_i + (1-lam) * y_j

Référence : https://github.com/basavyr/augmented-mixup-privacy-collaborative-training
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_c_factor(r: float, v_hat: float, d: int) -> float:
    """Mixup : c = r^2 / (2 * d * v_hat)."""
    return (r * r) / (2.0 * d * v_hat + 1e-20)


def mf_from_tau(alpha: float, tau: float, c: float) -> float:
    """Mixup : mf >= sqrt( (alpha^2/(tau(1-alpha)^2) - 1) / (2c) )."""
    a2 = alpha * alpha
    inner = max(a2 / (max(tau, 1e-20) * (1.0 - alpha) ** 2) - 1.0, 0.0)
    return float(np.sqrt(inner / (2.0 * max(c, 1e-20))))


def sample_ball_noise(shape: tuple, radius: float, device: torch.device) -> torch.Tensor:
    """Mixup : bruit uniforme sur la boule L2 de rayon donné (dernière dim = features)."""
    noise = torch.randn(*shape, device=device)
    norm = noise.norm(dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return noise / norm * radius


@torch.no_grad()
def estimate_embedding_stats(emb: torch.Tensor) -> tuple[float, float, int]:
    """Estime (r, v_hat, d) sur les embeddings de tokens aplatis [B, T, H] → [B, D]."""
    feats = emb.detach().float().reshape(emb.size(0), -1)
    n, d = feats.shape
    if n == 1:
        return 1.0, float(feats.var().item() + 1e-8), d
    perm = torch.randperm(n, device=feats.device)
    perm = torch.roll(perm, shifts=1)
    r = float(torch.norm(feats - feats[perm], p=2, dim=1).mean().item())
    mu = feats.mean()
    v_hat = float(((feats - mu) ** 2).mean().item())
    return max(r, 1e-6), max(v_hat, 1e-8), d


def resolve_radius(
    emb: torch.Tensor,
    alpha: float,
    tau: float,
    radius: float | None,
) -> float:
    """Utilise le rayon explicite s'il est donné ; sinon rayon = mf * r d'après le batch."""
    if radius is not None and radius > 0:
        return float(radius)
    r, v_hat, d = estimate_embedding_stats(emb)
    c = compute_c_factor(r=r, v_hat=v_hat, d=d)
    mf = mf_from_tau(alpha=alpha, tau=tau, c=c)
    return float(mf * r)


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    ignore = labels == -100
    safe = labels.clone()
    safe[ignore] = 0
    one_hot = F.one_hot(safe, num_classes=num_classes).float()
    one_hot[ignore] = 0.0
    return one_hot


def mixup_batch_token_embeddings(
    emb: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.7,
    tau: float = 1.0,
    radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mélange des embeddings de tokens.
    
        Retourne :
            mixed_emb [B, T, H], soft_targets [B, T, C], valid_mask [B, T]
        """
    batch_size, seq_len, hidden = emb.shape
    device = emb.device
    feats = emb.reshape(batch_size, -1)  # [B, D] like AMPCT flattened features
    feature_dim = feats.size(1)
    target_radius = resolve_radius(emb=emb, alpha=alpha, tau=tau, radius=radius)

    if batch_size == 1:
        # Cas limite : mélange avec soi-même + bruit
        lam = torch.tensor([0.5], device=device)
        noise = sample_ball_noise((1, feature_dim), target_radius, device=device)
        mix_feats = lam.view(1, 1) * feats + (1.0 - lam).view(1, 1) * (feats + noise)
        soft = labels_to_one_hot(labels=labels, num_classes=num_classes)
        valid = labels != -100
        return mix_feats.view(batch_size, seq_len, hidden), soft, valid

    # Appariement : décalage cyclique
    indices = torch.arange(batch_size, device=device)
    partner = torch.roll(indices, shifts=1)

    # lambda dans [0.3, 0.7]
    lam = 0.3 + 0.4 * torch.rand(batch_size, device=device)
    noise = sample_ball_noise((batch_size, feature_dim), target_radius, device=device)
    perturbed = feats[partner] + noise
    lam_view = lam.view(batch_size, 1)
    mix_feats = lam_view * feats + (1.0 - lam_view) * perturbed
    emb_mix = mix_feats.view(batch_size, seq_len, hidden)

    # Étiquettes souples (token), style mixup avec masque -100
    y_i = labels_to_one_hot(labels=labels, num_classes=num_classes)
    y_j = y_i[partner]
    lam_y = lam.view(batch_size, 1, 1)
    soft = lam_y * y_i + (1.0 - lam_y) * y_j
    valid_i = labels != -100
    valid_j = labels[partner] != -100
    valid_mask = valid_i | valid_j 
    soft = soft * valid_mask.unsqueeze(-1).float()
    return emb_mix, soft, valid_mask


# Noms rétrocompatibles
# utilisés par train.py
def singularized_mixup(
    emb: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.7,
    tau: float = 1.0,
    radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mixup_batch_token_embeddings(
        emb=emb,
        labels=labels,
        num_classes=num_classes,
        alpha=alpha,
        tau=tau,
        radius=radius,
    )


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft-CE au token (moyenne sur le batch ; on masque les tokens ignore)."""
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is not None:
        soft_targets = soft_targets * class_weights.view(1, 1, -1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    loss = loss.masked_select(valid_mask)
    if loss.numel() == 0:
        return logits.sum() * 0.0
    return loss.mean()
