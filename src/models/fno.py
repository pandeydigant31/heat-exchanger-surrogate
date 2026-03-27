"""Fourier Neural Operator for heat exchanger surrogate modeling.

The FNO learns a mapping from operating conditions (inlet temps, flow rates,
geometry) to temperature profiles T(x) along the heat exchanger.

Architecture (Li et al., 2021):
    Input → Lifting → [Fourier Layer × N] → Projection → Output

Each Fourier Layer:
    v = σ(W·v + K(v))  where K operates in Fourier space:
    K(v) = F⁻¹(R · F(v))  (pointwise multiplication of Fourier coefficients)

This is a lightweight PyTorch implementation — no PhysicsNeMo dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer: learns weights in frequency domain.

    Truncates to `modes` lowest frequency modes, applies learned
    complex-valued weights, then transforms back to physical space.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """x: (batch, channels, spatial_points)"""
        batch_size = x.shape[0]
        n = x.shape[2]

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        # Multiply low-frequency modes with learned weights
        out_ft = torch.zeros(batch_size, self.out_channels, n // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )

        # Inverse FFT
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FourierLayer(nn.Module):
    """Single Fourier layer: spectral convolution + linear transform + activation."""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.linear(x)))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for heat exchanger temperature profiles.

    Input: operating conditions (broadcast to spatial grid) → (batch, n_input, n_points)
    Output: temperature profiles → (batch, n_output, n_points)
    """

    def __init__(self, n_input, n_output, n_points=50,
                 width=64, modes=16, n_layers=4, dropout=0.1):
        super().__init__()
        self.n_points = n_points

        # Lifting: project input channels to hidden width
        self.lift = nn.Sequential(
            nn.Conv1d(n_input + 1, width, 1),  # +1 for spatial coordinate
            nn.GELU(),
        )

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes) for _ in range(n_layers)
        ])

        # Projection: hidden width → output channels
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, n_output, 1),
        )

    def forward(self, params):
        """
        Args:
            params: (batch, n_input) — operating conditions

        Returns:
            profiles: (batch, n_output, n_points) — temperature profiles
        """
        batch_size = params.shape[0]

        # Broadcast params to spatial grid
        # Shape: (batch, n_input, n_points)
        x = params.unsqueeze(-1).expand(-1, -1, self.n_points)

        # Append spatial coordinate as additional channel
        grid = torch.linspace(0, 1, self.n_points, device=params.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        x = torch.cat([x, grid], dim=1)

        # Lifting
        x = self.lift(x)

        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

        # Project to output
        return self.project(x)
