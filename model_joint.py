"""Transformer-based joint HRIR inpainting model with ITD/ILD auxiliary heads."""

import torch
import torch.nn as nn
import numpy as np


# =========================
# Model blocks
# =========================
class MicSelfAttentionBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask_obs):
        # mask_obs: (B, N) 1 observed, 0 missing
        key_padding_mask = (mask_obs == 0)  # True means "ignore" for attention
        x2, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(x2))
        x2 = self.ff(x)
        return self.norm2(x + self.drop(x2))


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2 ** torch.arange(num_freqs).float() * np.pi

    def forward(self, x):
        # x: (..., D)
        out = [x] if self.include_input else []
        for freq in self.freq_bands.to(x.device):
            out += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(out, dim=-1)


class PostDenoiser(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.denoise = nn.Conv1d(K, K, kernel_size=3, padding=1, groups=1)

    def forward(self, H_hat, mask):
        B, N, K = H_hat.shape
        smoothed = self.denoise(H_hat.transpose(1, 2)).transpose(1, 2)
        smoothed = torch.tanh(smoothed)
        mask_unsq = mask.unsqueeze(-1)
        return H_hat * mask_unsq + smoothed * (1 - mask_unsq)


# =========================
# InpaintNet + cue heads
# =========================
class InpaintNet(nn.Module):
    """
    Original transformer over directions + multi-head segment decoders over time (K).
    Added:
      - ITD head: predicts ITD (samples) per direction
      - ILD head: predicts ILD (dB) per band per direction

    ITD/ILD are NOT used in reconstruction in this version.
    """
    def __init__(
        self,
        K,
        d_model=256,
        n_layers=3,
        n_heads=4,
        num_freqs=6,
        dropout=0.1,
        ild_num_bands=12,
        itd_max_samples=32.0,
    ):
        super().__init__()
        self.geo_posenc = PositionalEncoding(num_freqs=num_freqs, include_input=True)

        # input is 42 because cat([geo_feat(3), geo_feat_pe(39)]) and geo_feat_pe includes input
        self.embed_geo = nn.Sequential(
            nn.Linear(42, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )


        self.rir_encoder = nn.Sequential(
            nn.Linear(K, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.blocks = nn.ModuleList([MicSelfAttentionBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.mask_emb = nn.Embedding(2, d_model)

        self.num_heads = n_heads
        assert K % self.num_heads == 0, f"K={K} must be divisible by num_heads={self.num_heads}"
        segment_len = int(K / self.num_heads)

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 512),
                    nn.GELU(),
                    nn.Linear(512, segment_len),
                )
                for _ in range(self.num_heads)
            ]
        )

        nn.init.normal_(self.mask_emb.weight, std=0.02)

        # Post-reconstruction denoiser
        self.denoiser = PostDenoiser(K)

        # -------------------------
        # Auxiliary ITD / ILD heads
        # -------------------------
        self.ild_num_bands = ild_num_bands
        self.itd_max_samples = float(itd_max_samples)

        self.itd_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.ild_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, ild_num_bands),
        )

    def forward(self, H_norm, mask, geo_feat, fuse=True, return_cues=False):
        """
        H_norm: (B, N, K)
        mask:   (B, N) 1 observed, 0 missing
        geo_feat: (B, N, 3)

        Returns:
          if return_cues=False:
              H_fused (B,N,K)
          else:
              H_fused, itd_pred (B,N), ild_pred_db (B,N,BANDS)
        """
        B, N, K = H_norm.shape

        H_masked = H_norm * mask.unsqueeze(-1)

        rir_feat = self.rir_encoder(H_masked.reshape(-1, K)).reshape(B, N, -1)

        geo_feat_pe = self.geo_posenc(geo_feat)  # (B,N,39)

        geo_emb = self.embed_geo(torch.cat([geo_feat, geo_feat_pe], dim=-1))  # (B,N,42)->(B,N,d)


        h = rir_feat + geo_emb

        for blk in self.blocks:
            h = blk(h, mask)

        dec_in = h  # (B,N,d)

        segments = [head(dec_in) for head in self.decoders]  # each (B,N,segment_len)
        H_hat_raw = torch.cat(segments, dim=-1)              # (B,N,K)

        if fuse:
            H_fused = H_norm * mask.unsqueeze(-1) + H_hat_raw * (1 - mask.unsqueeze(-1))
        else:
            H_fused = H_norm

        H_fused = self.denoiser(H_fused, mask)

        if not return_cues:
            return H_fused

        # -------------
        # cue predictions
        # -------------
        itd_pred = self.itd_head(h).squeeze(-1)  # (B,N)
        # bound it (in samples) for stability
        if self.itd_max_samples > 0:
            itd_pred = self.itd_max_samples * torch.tanh(itd_pred / self.itd_max_samples)

        ild_pred_db = self.ild_head(h)          # (B,N,BANDS)

        return H_fused, itd_pred, ild_pred_db
