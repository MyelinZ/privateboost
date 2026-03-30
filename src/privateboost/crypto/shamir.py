"""Shamir secret sharing over a finite field (m-of-n threshold).

All arithmetic is performed modulo a Mersenne prime (2^61 - 1).
Float values are encoded as fixed-point integers with configurable
precision before sharing and decoded back after reconstruction.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

# Mersenne prime 2^61 - 1
PRIME = (1 << 61) - 1

# Fixed-point precision: 2^24 gives ~7 decimal digits, leaves headroom
# for summing thousands of shares without overflow in intermediate products.
PRECISION = 1 << 24


@dataclass
class Share:
    """A Shamir share: evaluation point x and polynomial values y (mod p)."""

    x: int
    y: np.ndarray  # dtype=int64, values in [0, PRIME)


def _mul_mod(a: np.ndarray, b: int) -> np.ndarray:
    """Vectorized (a * b) mod PRIME using Python ints to avoid int64 overflow."""
    # Convert to Python ints, multiply, reduce — then back to int64 array
    result = np.array([int(v) * b % PRIME for v in a], dtype=np.int64)
    return result


def _add_mod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized (a + b) mod PRIME. Safe for int64 since a,b < 2^61."""
    s = a.astype(np.int64) + b.astype(np.int64)
    mask = s >= PRIME
    s[mask] -= PRIME
    mask2 = s < 0
    s[mask2] += PRIME
    return s


def _mod_inv(a: int, p: int) -> int:
    """Modular inverse via Fermat's little theorem."""
    return pow(int(a) % p, p - 2, p)


def _encode(values: np.ndarray) -> np.ndarray:
    """Encode float values as finite field elements (fixed-point)."""
    scaled = np.round(values * PRECISION).astype(np.int64)
    # Map negatives into [0, PRIME)
    result = np.array([int(v) % PRIME for v in scaled], dtype=np.int64)
    return result


def _decode(values: np.ndarray, count: int) -> np.ndarray:
    """Decode finite field elements back to floats."""
    half = PRIME // 2
    result = np.empty(count, dtype=np.float64)
    for i in range(count):
        v = int(values[i]) % PRIME
        if v > half:
            v -= PRIME
        result[i] = v / PRECISION
    return result


def share(
    values: np.ndarray,
    n_parties: int = 3,
    threshold: int = 2,
    rng: np.random.Generator | None = None,
) -> List[Share]:
    """Split a vector into Shamir shares over a finite field (m-of-n threshold)."""
    if threshold > n_parties:
        raise ValueError(f"Threshold {threshold} cannot exceed n_parties {n_parties}")

    n = len(values)
    encoded = _encode(values)

    # Random coefficients: uniform in [0, PRIME) for each degree 1..threshold-1
    if rng is None:
        rng = np.random.default_rng()
    # Use two int64 randoms to build a uniform value in [0, PRIME)
    rand_coeffs = []
    for _ in range(threshold - 1):
        hi = rng.integers(0, 1 << 30, size=n, dtype=np.int64)
        lo = rng.integers(0, 1 << 31, size=n, dtype=np.int64)
        coeff = np.array(
            [(int(h) << 31 | int(low)) % PRIME for h, low in zip(hi, lo)], dtype=np.int64
        )
        rand_coeffs.append(coeff)

    # Evaluate polynomial at x = 1, 2, ..., n_parties
    shares_list = []
    for x in range(1, n_parties + 1):
        y = encoded.copy()
        x_power = 1
        for k in range(threshold - 1):
            x_power = (x_power * x) % PRIME
            y = _add_mod(y, _mul_mod(rand_coeffs[k], x_power))
        shares_list.append(Share(x=x, y=y))

    return shares_list


def reconstruct(shares: List[Share], threshold: int = 2) -> np.ndarray:
    """Reconstruct a vector from Shamir shares using Lagrange interpolation."""
    if len(shares) < threshold:
        raise ValueError(f"Need at least {threshold} shares to reconstruct, got {len(shares)}")

    shares = shares[:threshold]
    n = len(shares[0].y)
    x_values = [s.x for s in shares]

    # Lagrange coefficients at x=0
    coeffs = []
    for j in range(threshold):
        num = 1
        den = 1
        for i in range(threshold):
            if i != j:
                num = (num * (-x_values[i])) % PRIME
                den = (den * (x_values[j] - x_values[i])) % PRIME
        coeffs.append((num * _mod_inv(den, PRIME)) % PRIME)

    # Interpolate: result = sum(coeff_i * y_i) mod PRIME
    result = np.zeros(n, dtype=np.int64)
    for i in range(threshold):
        result = _add_mod(result, _mul_mod(shares[i].y, coeffs[i]))

    return _decode(result, n)


def mod_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise modular addition of share vectors."""
    return _add_mod(a, b)
