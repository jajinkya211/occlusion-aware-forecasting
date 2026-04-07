import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return self.output_norm(x[:, -1, :])


class SpatialAttentionLayer(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + residual
        return self.feed_forward(self.norm2(x)) + x


class VisibilityFiLM(nn.Module):

    def __init__(self, hidden_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
        )
        self.gamma_projection = nn.Linear(embed_dim * 2, hidden_dim)
        self.beta_projection  = nn.Linear(embed_dim * 2, hidden_dim)
        nn.init.zeros_(self.gamma_projection.weight)
        nn.init.ones_(self.gamma_projection.bias)
        nn.init.zeros_(self.beta_projection.weight)
        nn.init.zeros_(self.beta_projection.bias)

    def forward(self, h: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, hidden_dim = h.shape
        film_embed = self.mlp(visibility.view(batch_size * num_agents, 1))
        gamma = self.gamma_projection(film_embed).view(batch_size, num_agents, hidden_dim)
        beta  = self.beta_projection(film_embed).view(batch_size, num_agents, hidden_dim)
        return gamma * h + beta


class MultiModalPredictionHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        pred_len: int,
        num_modes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.pred_len = pred_len

        self.mode_logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modes),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_modes * pred_len * 2),
        )
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modes * pred_len * 2),
        )
        self.epistemic_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.epistemic_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        agent_embeddings: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_agents, _ = agent_embeddings.shape
        K, T = self.num_modes, self.pred_len

        trajectories = self.trajectory_head(agent_embeddings).view(batch_size, num_agents, K, T, 2)

        log_var_aleatoric = (
            self.aleatoric_head(agent_embeddings)
            .view(batch_size, num_agents, K, T, 2)
            .clamp(-4.0, 1.0)
        )

        occlusion = (1.0 - visibility).clamp(0.0, 1.0).view(batch_size * num_agents, 1)
        epistemic_scale = self.epistemic_head(occlusion).view(batch_size, num_agents, 1, 1, 1)

        epistemic_weight = self.epistemic_weight.clamp(min=0.0, max=3.0)
        log_var_total = log_var_aleatoric + epistemic_weight * torch.log1p(epistemic_scale)

        return {
            'trajectories':   trajectories,
            'log_variances':  log_var_total,
            'mode_logits':    self.mode_logits_head(agent_embeddings),
            'epistemic_scale': epistemic_scale.view(batch_size, num_agents),
        }


class OcclusionAwareForecaster(nn.Module):

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        pred_len: int = 6,
        num_modes: int = 6,
        num_heads: int = 4,
        num_temporal_layers: int = 2,
        num_spatial_layers: int = 2,
        film_embed_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, num_heads, num_temporal_layers, dropout
        )
        self.film_pre_social  = VisibilityFiLM(hidden_dim, film_embed_dim)
        self.film_post_social = VisibilityFiLM(hidden_dim, film_embed_dim)
        self.spatial_layers   = nn.ModuleList([
            SpatialAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_spatial_layers)
        ])
        self.prediction_head = MultiModalPredictionHead(
            hidden_dim, pred_len, num_modes, dropout
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        past_trajectory: torch.Tensor,
        visibility: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, obs_len, num_agents, input_dim = past_trajectory.shape

        agent_seqs = past_trajectory.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents, obs_len, input_dim
        )
        agent_embeddings = self.temporal_encoder(agent_seqs).view(
            batch_size, num_agents, self.hidden_dim
        )
        agent_embeddings = self.film_pre_social(agent_embeddings, visibility)

        padding_mask = ~agent_mask
        for layer in self.spatial_layers:
            agent_embeddings = layer(agent_embeddings, key_padding_mask=padding_mask)

        agent_embeddings = self.film_post_social(agent_embeddings, visibility)
        agent_embeddings = agent_embeddings * agent_mask.unsqueeze(-1).float()

        return self.prediction_head(agent_embeddings, visibility)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoFiLMForecaster(OcclusionAwareForecaster):

    def forward(
        self,
        past_trajectory: torch.Tensor,
        visibility: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, obs_len, num_agents, input_dim = past_trajectory.shape
        agent_seqs = past_trajectory.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents, obs_len, input_dim
        )
        agent_embeddings = self.temporal_encoder(agent_seqs).view(
            batch_size, num_agents, self.hidden_dim
        )
        padding_mask = ~agent_mask
        for layer in self.spatial_layers:
            agent_embeddings = layer(agent_embeddings, key_padding_mask=padding_mask)
        agent_embeddings = agent_embeddings * agent_mask.unsqueeze(-1).float()
        return self.prediction_head(agent_embeddings, visibility)
