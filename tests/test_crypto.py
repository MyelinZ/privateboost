"""Tests for cryptographic primitives: Shamir sharing and commitments."""

import numpy as np

from privateboost.crypto import Share, compute_commitment, reconstruct, share


def test_shamir_reconstruction():
    """Test that Shamir sharing and reconstruction work correctly."""
    np.random.seed(42)

    values = np.array([42.0, 100.0, -7.5])
    shares = share(values, n_parties=3, threshold=2)

    # Reconstruct from different pairs
    reconstructed_01 = reconstruct([shares[0], shares[1]], threshold=2)
    reconstructed_02 = reconstruct([shares[0], shares[2]], threshold=2)
    reconstructed_12 = reconstruct([shares[1], shares[2]], threshold=2)

    np.testing.assert_array_almost_equal(reconstructed_01, values)
    np.testing.assert_array_almost_equal(reconstructed_02, values)
    np.testing.assert_array_almost_equal(reconstructed_12, values)


def test_shamir_linearity():
    """Test that Shamir sharing is additively homomorphic."""
    np.random.seed(42)

    values_a = np.array([10.0, 20.0, 30.0])
    values_b = np.array([1.0, 2.0, 3.0])

    shares_a = share(values_a, n_parties=3, threshold=2)
    shares_b = share(values_b, n_parties=3, threshold=2)

    # Sum shares at each shareholder
    summed_shares = [
        Share(x=shares_a[i].x, y=shares_a[i].y + shares_b[i].y)
        for i in range(3)
    ]

    # Reconstruct should give sum of values
    reconstructed = reconstruct([summed_shares[0], summed_shares[1]], threshold=2)

    np.testing.assert_array_almost_equal(reconstructed, values_a + values_b)


def test_commitment_consistency():
    """Test that commitments are consistent for same inputs."""
    commitment1 = compute_commitment(round_id=0, client_id="client_1", nonce=b"x" * 32)
    commitment2 = compute_commitment(round_id=0, client_id="client_1", nonce=b"x" * 32)
    commitment3 = compute_commitment(round_id=0, client_id="client_1", nonce=b"y" * 32)

    assert commitment1 == commitment2
    assert commitment1 != commitment3


def test_commitment_round_separation():
    """Test that different rounds produce different commitments."""
    nonce = b"z" * 32
    commitment_r0 = compute_commitment(round_id=0, client_id="client_1", nonce=nonce)
    commitment_r1 = compute_commitment(round_id=1, client_id="client_1", nonce=nonce)

    assert commitment_r0 != commitment_r1


def test_3_of_5_threshold():
    """Test 3-of-5 threshold scheme."""
    np.random.seed(42)

    values = np.array([42.0, 100.0])
    shares = share(values, n_parties=5, threshold=3)

    reconstructed = reconstruct([shares[0], shares[2], shares[4]], threshold=3)
    np.testing.assert_array_almost_equal(reconstructed, values)

    reconstructed2 = reconstruct([shares[1], shares[3], shares[4]], threshold=3)
    np.testing.assert_array_almost_equal(reconstructed2, values)
