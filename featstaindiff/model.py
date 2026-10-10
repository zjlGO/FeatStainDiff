"""Paper-aligned CSB, U-ViT denoiser, and frequency-domain experts.

The model operates on paired *feature vectors*, not pixels. Model sizes and the
kernel bandwidth heuristic are configurable release defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch_dct


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 1024
    hidden_dim: int = 384
    depth: int = 6
    heads: int = 8
    patch_size: int = 16
    semantic_depth: int = 3
    num_experts: int = 4
    contrastive_temperature: float = 0.1
    mmd_bandwidth: float | None = None

    def __post_init__(self) -> None:
        if self.feature_dim <= 0 or self.patch_size <= 0 or self.feature_dim % self.patch_size:
            raise ValueError("feature_dim must be a positive multiple of patch_size")
        if self.hidden_dim < 2 or self.heads <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be at least 2 and divisible by positive heads")
        if self.depth < 2 or self.depth % 2:
            raise ValueError("depth must be even and at least 2")
        if self.semantic_depth < self.depth // 2:
            raise ValueError("semantic_depth must cover every U-ViT input block")
        if self.num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not math.isfinite(self.contrastive_temperature) or self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be finite and positive")
        if self.mmd_bandwidth is not None and (
            not math.isfinite(self.mmd_bandwidth) or self.mmd_bandwidth <= 0
        ):
            raise ValueError("mmd_bandwidth must be finite and positive")


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of continuous diffusion time in [0, 1]."""
    half = dim // 2
    if half == 0:
        raise ValueError("hidden dimension must be at least 2")
    scales = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    phase = (t.float() * 999)[:, None] * scales[None]
    out = torch.cat((phase.cos(), phase.sin()), dim=-1)
    return F.pad(out, (0, dim - out.shape[-1]))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, with_skip: bool = False) -> None:
        super().__init__()
        self.skip = nn.Linear(dim * 2, dim) if with_skip else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        if self.skip is not None:
            if skip is None:
                raise ValueError("a U-ViT output block requires its skip connection")
            x = self.skip(torch.cat((x, skip), dim=-1))
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SemanticEncoder(nn.Module):
    """Patchify a feature vector and aggregate it through a [SEM] token."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        tokens = cfg.feature_dim // cfg.patch_size
        self.patch_embed = nn.Conv1d(1, cfg.hidden_dim, cfg.patch_size, cfg.patch_size)
        self.sem_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, cfg.hidden_dim))
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.hidden_dim, cfg.heads) for _ in range(cfg.semantic_depth)
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        nn.init.trunc_normal_(self.sem_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, vectors: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        patches = self.patch_embed(vectors[:, None, :]).transpose(1, 2)
        sem = self.sem_token.expand(vectors.shape[0], -1, -1)
        x = torch.cat((sem, patches), dim=1) + self.position
        stages = []
        for block in self.blocks:
            x = block(x)
            stages.append(x[:, 1:, :])
        x = self.norm(x)
        return x[:, 0, :], x[:, 1:, :], stages


class FrequencyExpert(nn.Module):
    """DCT -> learned spectral filter -> MHSA -> inverse DCT."""

    def __init__(self, tokens: int, dim: int, heads: int) -> None:
        super().__init__()
        self.filter = nn.Parameter(torch.ones(1, tokens, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        # torch-dct computes along the final axis; transform the token axis.
        spectral = torch_dct.dct(x.float().transpose(1, 2), norm="ortho").transpose(1, 2)
        spectral = (spectral * self.filter).to(dtype=x.dtype)
        h = self.norm(spectral)
        spectral = spectral + self.attn(h, h, h, need_weights=False)[0]
        return torch_dct.idct(spectral.float().transpose(1, 2), norm="ortho").transpose(1, 2).to(x.dtype)


class FrequencyMoE(nn.Module):
    """Time-conditioned, globally routed frequency experts."""

    def __init__(self, tokens: int, dim: int, heads: int, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(FrequencyExpert(tokens, dim, heads) for _ in range(num_experts))
        self.feature_gate = nn.Linear(dim, dim)
        self.time_gate = nn.Linear(dim, dim)
        self.gate_out = nn.Linear(dim, num_experts)
        self.exploration = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x: Tensor, time_embedding: Tensor) -> tuple[Tensor, Tensor]:
        pooled = x.mean(dim=1) + x.amax(dim=1)
        logits = self.gate_out(F.gelu(self.feature_gate(pooled) + self.time_gate(time_embedding)))
        weights = (logits + self.exploration).softmax(dim=-1)
        result = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            result = result + weights[:, index, None, None] * expert(x)
        return result, weights


def contrastive_loss(he_sem: Tensor, ihc_sem: Tensor, temperature: float) -> Tensor:
    """H&E-anchored InfoNCE with matching IHC features on the diagonal."""
    if he_sem.shape[0] < 2:
        return he_sem.new_zeros(())
    he_sem = F.normalize(he_sem.float(), dim=-1)
    ihc_sem = F.normalize(ihc_sem.float(), dim=-1)
    logits = he_sem @ ihc_sem.T / temperature
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


def gaussian_mmd(he_tokens: Tensor, ihc_tokens: Tensor, bandwidth: float | None) -> Tensor:
    """Biased Gaussian MMD² on non-[SEM] tokens, averaged over paired samples.

    A detached median squared cross-domain distance determines sigma² when no
    bandwidth is supplied. This avoids assuming an unpublished feature scale.
    """
    he_tokens, ihc_tokens = he_tokens.float(), ihc_tokens.float()
    xx = torch.cdist(he_tokens, he_tokens).square()
    yy = torch.cdist(ihc_tokens, ihc_tokens).square()
    xy = torch.cdist(he_tokens, ihc_tokens).square()
    sigma2 = xy.detach().flatten().median().clamp_min(1e-8) if bandwidth is None else bandwidth**2
    return ((-xx / (2 * sigma2)).exp().mean((1, 2))
            + (-yy / (2 * sigma2)).exp().mean((1, 2))
            - 2 * (-xy / (2 * sigma2)).exp().mean((1, 2))).mean()


def load_balance_loss(weights: Tensor) -> Tensor:
    """Squared CV of mean expert utilization across the training batch."""
    utilization = weights.float().mean(dim=0)
    return utilization.var(unbiased=False) / utilization.mean().square().clamp_min(1e-8)


class FeatStainDiff(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        tokens = config.feature_dim // config.patch_size
        self.he_encoder = SemanticEncoder(config)
        self.ihc_encoder = SemanticEncoder(config)
        self.patch_embed = nn.Conv1d(2, config.hidden_dim, config.patch_size, config.patch_size)
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, tokens + 1, config.hidden_dim))
        self.semantic_skips = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.depth // 2)
        )
        self.in_blocks = nn.ModuleList(
            TransformerBlock(config.hidden_dim, config.heads) for _ in range(config.depth // 2)
        )
        self.mid_before = TransformerBlock(config.hidden_dim, config.heads)
        self.fmoe = FrequencyMoE(tokens + 1, config.hidden_dim, config.heads, config.num_experts)
        self.mid_after = TransformerBlock(config.hidden_dim, config.heads)
        self.out_blocks = nn.ModuleList(
            TransformerBlock(config.hidden_dim, config.heads, with_skip=True)
            for _ in range(config.depth // 2)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, config.patch_size)
        nn.init.trunc_normal_(self.position, std=0.02)

    def prepare_condition(self, he: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Cache the H&E semantic sequence across DDIM steps at inference."""
        if he.ndim != 2 or he.shape[1] != self.config.feature_dim:
            raise ValueError(f"he must have shape [batch, {self.config.feature_dim}]")
        return self.he_encoder(he)

    def forward(
        self,
        noisy_ihc: Tensor,
        he: Tensor,
        t: Tensor,
        target_ihc: Tensor | None = None,
        *,
        condition: tuple[Tensor, Tensor, list[Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config
        if noisy_ihc.ndim != 2 or he.shape != noisy_ihc.shape or noisy_ihc.shape[1] != cfg.feature_dim:
            raise ValueError(f"noisy_ihc and he must have shape [batch, {cfg.feature_dim}]")
        if t.shape != (he.shape[0],):
            raise ValueError("t must have shape [batch]")
        he_sem, he_tokens, he_stages = (
            self.prepare_condition(he) if condition is None else condition
        )
        paired = torch.stack((noisy_ihc, he), dim=1)
        patches = self.patch_embed(paired).transpose(1, 2)
        time_embedding = self.time_embed(timestep_embedding(t, cfg.hidden_dim))
        x = torch.cat((time_embedding[:, None, :], patches), dim=1) + self.position
        skips = []
        for block, projection, stage in zip(self.in_blocks, self.semantic_skips, he_stages):
            x = torch.cat((x[:, :1, :], x[:, 1:, :] + projection(stage)), dim=1)
            x = block(x)
            skips.append(x)
        x = self.mid_before(x)
        x, gates = self.fmoe(x, time_embedding)
        x = self.mid_after(x)
        for block in self.out_blocks:
            x = block(x, skips.pop())
        pred_noise = self.decoder(self.norm(x)[:, 1:, :]).reshape(he.shape[0], cfg.feature_dim)
        result = {"pred_noise": pred_noise, "gates": gates}
        if target_ihc is not None:
            if target_ihc.shape != he.shape:
                raise ValueError("target_ihc must match he shape")
            ihc_sem, ihc_tokens, _ = self.ihc_encoder(target_ihc)
            result["contrastive"] = contrastive_loss(he_sem, ihc_sem, cfg.contrastive_temperature)
            result["mmd"] = gaussian_mmd(he_tokens, ihc_tokens, cfg.mmd_bandwidth)
            result["load"] = load_balance_loss(gates)
        return result
