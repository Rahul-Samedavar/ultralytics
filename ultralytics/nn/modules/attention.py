import torch.nn as nn
import torch


class StAtn(nn.Module):
    def __init__(self, kernel_size, stride,learn_alpha=True,learn_tau=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=learn_alpha)
        tau = torch.tensor(3.0)
        self.log_tau =nn.Parameter(torch.log(tau), requires_grad=learn_tau)
        self.b = kernel_size // 2
        self.pool = nn.AvgPool2d(kernel_size, stride)

    def mask_borders(self, x):
        mask = torch.ones_like(x)
        mask[:, :, :self.b, :] = 0
        mask[:, :, -self.b:, :] = 0
        mask[:, :, :, :self.b] = 0
        mask[:, :, :, -self.b:] = 0
        return x * mask


    def forward(self, x: list[torch.Tensor]):
        if len(x) != 2:
            raise ValueError(f"expected x to be [x, attn_map] but got list of size {len(x)}")
        x, attn_map = x

        attn_map = self.mask_borders(attn_map)
        attn_map = self.pool(attn_map)

        # print(f"X: {x.shape}")
        # print(f"Atn: {attn_map.shape}")
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        tau = torch.exp(self.log_tau).clamp(1.5, 6.0)
        attn = torch.sigmoid(attn_map / tau)
        return x * (1 + alpha*attn)

import torch.nn.functional as F
import cv2
import numpy as np

class TextureAnomalyMap(nn.Module):
    def __init__(self):
        super().__init__()

        kernels = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            k = cv2.getGaborKernel(
                (15, 15),
                sigma=3.0,
                theta=theta,
                lambd=6.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            k /= np.linalg.norm(k)
            kernels.append(k)

        gabor = torch.from_numpy(np.stack(kernels)).unsqueeze(1)  # (4,1,15,15)
        self.register_buffer("gabor", gabor)

        sobel_x = torch.tensor(
            [[-1,  0,  1],
             [-2,  0,  2],
             [-1,  0,  1]],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

        self.requires_grad_(False)

    def forward(self, x):
        """
        x : [B, C, H, W]  (YOLO-style input)
        returns : [B, 1, H, W] the texture anomaky map
        """

        if x.shape[1] > 1:
            x_gray = x.mean(dim=1, keepdim=True)
        else:
            x_gray = x

        fine = F.conv2d(x_gray, self.gabor, padding=7)   
        fine_energy = fine.abs().mean(dim=1, keepdim=True)

        coarse_small = F.avg_pool2d(fine_energy, 11, 1, 5)
        coarse_large = F.avg_pool2d(fine_energy, 31, 1, 15)

        anomaly = fine_energy - 0.5 * coarse_small - 0.5 * coarse_large
        anomaly = anomaly.clamp(min=0)

        grad_x = torch.abs(F.conv2d(x_gray, self.sobel_x, padding=1))
        grad_y = torch.abs(F.conv2d(x_gray, self.sobel_y, padding=1))
        edge_strength = grad_x + grad_y

        anomaly = anomaly * torch.exp(-edge_strength)

        max_val = anomaly.amax(dim=(2, 3), keepdim=True) + 1e-6
        anomaly = anomaly / max_val
        anomaly = anomaly.clamp(0, 3) / 3.0
        return anomaly

import math
from torchvision.transforms.functional import rgb_to_grayscale

class ScribbleMap(nn.Module):
    def __init__(self, num_orient=8, sharpness=10.0):
        super().__init__()
        self.num_orient = num_orient
        self.sharpness = sharpness

        angles = torch.linspace(0, math.pi, num_orient + 1)[:-1]
        kernels = []

        for a in angles:
            dx, dy = torch.cos(a), torch.sin(a)
            k = torch.zeros(1, 1, 3, 3)

            k[0,0,0,0] = -dx
            k[0,0,0,2] =  dx
            k[0,0,2,0] = -dx
            k[0,0,2,2] =  dx
            k[0,0,0,1] = -dy
            k[0,0,2,1] =  dy
            # center = 0 → zero-sum filter

            kernels.append(k)

        self.register_buffer(
            "orient_kernels",
            torch.cat(kernels, dim=0)  # (8,1,3,3)
        )

        # --- Laplacian kernel ---
        self.register_buffer(
            "lap_k",
            torch.tensor([[[[0,-1,0],
                            [-1,4,-1],
                            [0,-1,0]]]], dtype=torch.float32)
        )

    def forward(self, img):
        """
        img: B × 3 × H × W
        """


        if img.shape[1] > 1:
            img = rgb_to_grayscale(img, num_output_channels=1)



        # Orientation responses
        grad = F.conv2d(img, self.orient_kernels, padding=1)  # B×8×H×W
        orient_stack = grad.abs()

        # Mean + variance across orientations
        orient_mean = orient_stack.mean(dim=1, keepdim=True)
        orient_var  = orient_stack.var(dim=1, keepdim=True)

        # Connectivity (Laplacian)
        lap = F.conv2d(img, self.lap_k, padding=1).abs()
        connectivity = F.max_pool2d(lap, kernel_size=5, stride=1, padding=2)

        # Soft gating instead of hard threshold
        orient_gate = torch.sigmoid((orient_mean - 0.3) * self.sharpness)

        score = orient_var * connectivity * orient_gate

        # Normalize to [0,1] (stable)
        score = score / (score.amax(dim=[2,3], keepdim=True) + 1e-6)

        return score