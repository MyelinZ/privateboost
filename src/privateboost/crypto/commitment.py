"""Commitment scheme for client anonymity."""

import hashlib
import secrets
import struct


def generate_nonce() -> bytes:
    """Generate a fresh 32-byte random nonce."""
    return secrets.token_bytes(32)


def compute_commitment(round_id: int, client_id: str, nonce: bytes) -> bytes:
    """Compute commitment hash.

    commitment = SHA256(round_id || client_id || nonce)

    The commitment is opaque - aggregator cannot reverse it to learn client_id.

    Args:
        round_id: Integer identifying the protocol round.
        client_id: Client's internal identifier.
        nonce: 32-byte random value, fresh each round.

    Returns:
        32-byte SHA256 hash.
    """
    h = hashlib.sha256()
    h.update(struct.pack(">Q", round_id))  # 8-byte big-endian
    h.update(client_id.encode("utf-8"))
    h.update(nonce)
    return h.digest()
