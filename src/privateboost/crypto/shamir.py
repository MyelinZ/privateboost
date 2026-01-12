"""Shamir secret sharing (m-of-n threshold)."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Share:
    """A Shamir share: evaluation point x and polynomial value y."""

    x: int
    y: np.ndarray


def _lagrange_coefficients_at_zero(x_values: np.ndarray) -> np.ndarray:
    """Compute Lagrange basis coefficients at x=0: L_j(0) = prod_{i!=j} -x_i / (x_j - x_i)."""
    n = len(x_values)
    coeffs = np.ones(n)
    for j in range(n):
        for i in range(n):
            if i != j:
                coeffs[j] *= -x_values[i] / (x_values[j] - x_values[i])
    return coeffs


def share(values: np.ndarray, n_parties: int = 3, threshold: int = 2) -> List[Share]:
    """Split a vector into Shamir shares (m-of-n threshold).

    Each element gets independent random coefficients for security.
    """
    if threshold > n_parties:
        raise ValueError(f"Threshold {threshold} cannot exceed n_parties {n_parties}")

    coeffs = np.vstack([
        values,
        np.random.uniform(-1e10, 1e10, size=(threshold - 1, len(values)))
    ])
    x_points = np.arange(1, n_parties + 1)
    vander = np.vander(x_points, N=threshold, increasing=True)
    y_matrix = vander @ coeffs

    return [Share(x=int(x), y=y_matrix[i]) for i, x in enumerate(x_points)]


def reconstruct(shares: List[Share], threshold: int = 2) -> np.ndarray:
    """Reconstruct a vector from Shamir shares using Lagrange interpolation."""
    if len(shares) < threshold:
        raise ValueError(
            f"Need at least {threshold} shares to reconstruct, got {len(shares)}"
        )

    shares = shares[:threshold]
    x_values = np.array([s.x for s in shares])
    y_matrix = np.array([s.y for s in shares])

    return _lagrange_coefficients_at_zero(x_values) @ y_matrix
