use privateboost::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn test_shamir_reconstruction() {
    let mut rng = StdRng::seed_from_u64(42);
    let values = encode_all(&[42.0, 100.0, -7.5]);
    let shares = share(&values, 3, 2, &mut rng).unwrap();

    let r01 = decode_all(&reconstruct(&[shares[0].clone(), shares[1].clone()], 2).unwrap());
    let r02 = decode_all(&reconstruct(&[shares[0].clone(), shares[2].clone()], 2).unwrap());
    let r12 = decode_all(&reconstruct(&[shares[1].clone(), shares[2].clone()], 2).unwrap());

    for (i, exp) in [42.0, 100.0, -7.5].iter().enumerate() {
        assert!((r01[i] - exp).abs() < 1e-5);
        assert!((r02[i] - exp).abs() < 1e-5);
        assert!((r12[i] - exp).abs() < 1e-5);
    }
}

#[test]
fn test_shamir_linearity() {
    let mut rng = StdRng::seed_from_u64(42);
    let a = encode_all(&[10.0, 20.0, 30.0]);
    let b = encode_all(&[1.0, 2.0, 3.0]);

    let sa = share(&a, 3, 2, &mut rng).unwrap();
    let sb = share(&b, 3, 2, &mut rng).unwrap();

    let summed: Vec<Share> = sa.iter().zip(sb.iter()).map(|(a, b)| a.add(b)).collect();
    let result = decode_all(&reconstruct(&[summed[0].clone(), summed[1].clone()], 2).unwrap());

    for (i, exp) in [11.0, 22.0, 33.0].iter().enumerate() {
        assert!((result[i] - exp).abs() < 1e-5);
    }
}

#[test]
fn test_commitment_consistency() {
    let c1 = commit(0, "client_1", &[b'x'; 32]);
    let c2 = commit(0, "client_1", &[b'x'; 32]);
    let c3 = commit(0, "client_1", &[b'y'; 32]);
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
}

#[test]
fn test_commitment_round_separation() {
    let nonce = [b'z'; 32];
    let c0 = commit(0, "client_1", &nonce);
    let c1 = commit(1, "client_1", &nonce);
    assert_ne!(c0, c1);
}

#[test]
fn test_3_of_5_threshold() {
    let mut rng = StdRng::seed_from_u64(42);
    let values = encode_all(&[42.0, 100.0]);
    let shares = share(&values, 5, 3, &mut rng).unwrap();

    let r1 = decode_all(
        &reconstruct(
            &[shares[0].clone(), shares[2].clone(), shares[4].clone()],
            3,
        )
        .unwrap(),
    );
    let r2 = decode_all(
        &reconstruct(
            &[shares[1].clone(), shares[3].clone(), shares[4].clone()],
            3,
        )
        .unwrap(),
    );

    for (i, exp) in [42.0, 100.0].iter().enumerate() {
        assert!((r1[i] - exp).abs() < 1e-5);
        assert!((r2[i] - exp).abs() < 1e-5);
    }
}
