
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGF(nn.Module):
    """Image-controlled prior fusion"""
    def __init__(self, channels, prior_channels=1):
        super().__init__()

        self.prior_proj = nn.Sequential(
            nn.Conv2d(prior_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: list[torch.Tensor]):
        """
        F_img   : [B, C, H, W]  image features (YOLO backbone)
        M_prior : [B, 1, H0, W0] prior mask (any resolution)
        """

        if len(x) != 2:
            raise ValueError(f"expected x to be [x, attn_map] but got list of size {len(x)}")

        F_img, M_prior = x
        # Resize prior to feature resolution
        M = F.interpolate(
            M_prior,
            size=F_img.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        P = self.prior_proj(M)

        g = self.gate(F_img)

        F_out = F_img + self.scale * g * P

        return F_out

