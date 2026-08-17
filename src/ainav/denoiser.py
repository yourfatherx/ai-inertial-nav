"""Learning-based gyro denoiser (Phase 3).

Idea (following Brossard et al. 2020, "Denoising IMU Gyroscopes with Deep
Learning"): a small 1-D convolutional network looks at a window of raw IMU
(gyro + accel, 6 channels) and predicts a tiny *correction* to add to the raw
gyro at each timestep:

    w_corrected[t] = w_raw[t] + c[t],   c[t] = alpha * tanh(net(window)[t])

The correction is deliberately small (bounded by `alpha`, rad/s) because the
raw gyro is already close -- the network only has to shave off slowly varying
bias and noise, not reinvent the signal. Bounding with tanh keeps early
training stable (a random net can't inject huge spurious rates).

Dilated convolutions give a wide receptive field (past+future context around
each sample) with few parameters -- important on a 6 GB laptop GPU.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GyroDenoiser(nn.Module):
    def __init__(self, in_ch: int = 6, hidden: int = 32,
                 dilations=(1, 2, 4, 8, 16, 32), alpha: float = 0.05):
        """
        in_ch:     input channels (3 gyro + 3 accel).
        hidden:    conv width.
        dilations: one residual conv block per entry; receptive field grows
                   as 1 + 2*sum(dilations)*(k-1)/... -- with k=3 and these
                   dilations the net sees ~126 samples each side (~0.6 s @200Hz).
        alpha:     max correction magnitude in rad/s (tanh-bounded).
        """
        super().__init__()
        self.alpha = alpha
        k = 3
        layers = [nn.Conv1d(in_ch, hidden, k, padding=1), nn.GELU()]
        for d in dilations:
            layers += [
                nn.Conv1d(hidden, hidden, k, padding=d, dilation=d),
                nn.GELU(),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 3, 1)  # -> 3 gyro correction channels
        # Start near-zero so the initial correction ~= 0 (net == raw gyro).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, T) raw IMU window -> correction (B, 3, T) in rad/s."""
        h = self.backbone(x)
        c = self.head(h)
        return self.alpha * torch.tanh(c)

    def correct_gyro(self, imu: torch.Tensor) -> torch.Tensor:
        """Convenience: (B,6,T) raw IMU -> (B,3,T) corrected gyro."""
        gyro = imu[:, :3, :]
        return gyro + self.forward(imu)
