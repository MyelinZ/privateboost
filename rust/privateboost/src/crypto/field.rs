use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distr::{Distribution, StandardUniform};

pub const PRIME: u64 = (1u64 << 61) - 1; // 2^61 - 1 = 2305843009213693951

pub const ZERO: MersenneField = MersenneField(0);
pub const ONE: MersenneField = MersenneField(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct MersenneField(u64);

pub type F = MersenneField;

fn reduce(x: u128) -> u64 {
    let lo = (x as u64) & PRIME; // Lower 61 bits
    let hi = (x >> 61) as u64; // Upper bits
    let sum = lo + hi;
    if sum >= PRIME { sum - PRIME } else { sum }
}

impl MersenneField {
    pub fn from_u64(v: u64) -> MersenneField {
        MersenneField(v % PRIME)
    }

    pub fn from_i64(v: i64) -> MersenneField {
        if v >= 0 {
            MersenneField(v as u64 % PRIME)
        } else {
            // v is negative; add PRIME to bring into range
            let abs = (-v) as u64 % PRIME;
            if abs == 0 {
                MersenneField(0)
            } else {
                MersenneField(PRIME - abs)
            }
        }
    }

    pub fn to_i64(&self) -> i64 {
        let half = PRIME / 2;
        if self.0 > half {
            self.0 as i64 - PRIME as i64
        } else {
            self.0 as i64
        }
    }

    pub fn inner(&self) -> u64 {
        self.0
    }

    /// Multiplicative inverse via Fermat's little theorem: a^(p-2) mod p.
    /// Returns None for zero.
    pub fn inverse(&self) -> Option<MersenneField> {
        if self.0 == 0 {
            return None;
        }
        // Binary exponentiation: compute self^(PRIME-2)
        let mut base = *self;
        let mut exp = PRIME - 2;
        let mut result = ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            base = base * base;
            exp >>= 1;
        }
        Some(result)
    }
}

impl Add for MersenneField {
    type Output = MersenneField;
    fn add(self, other: MersenneField) -> MersenneField {
        MersenneField(reduce((self.0 as u128) + (other.0 as u128)))
    }
}

impl Sub for MersenneField {
    type Output = MersenneField;
    fn sub(self, other: MersenneField) -> MersenneField {
        if self.0 >= other.0 {
            MersenneField(self.0 - other.0)
        } else {
            MersenneField(PRIME - other.0 + self.0)
        }
    }
}

impl Mul for MersenneField {
    type Output = MersenneField;
    fn mul(self, other: MersenneField) -> MersenneField {
        MersenneField(reduce((self.0 as u128) * (other.0 as u128)))
    }
}

impl Neg for MersenneField {
    type Output = MersenneField;
    fn neg(self) -> MersenneField {
        if self.0 == 0 {
            MersenneField(0)
        } else {
            MersenneField(PRIME - self.0)
        }
    }
}

impl AddAssign for MersenneField {
    fn add_assign(&mut self, other: MersenneField) {
        *self = *self + other;
    }
}

impl SubAssign for MersenneField {
    fn sub_assign(&mut self, other: MersenneField) {
        *self = *self - other;
    }
}

impl MulAssign for MersenneField {
    fn mul_assign(&mut self, other: MersenneField) {
        *self = *self * other;
    }
}

impl fmt::Display for MersenneField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Distribution<MersenneField> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MersenneField {
        MersenneField(rng.random::<u64>() % PRIME)
    }
}

// FieldVec

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldVec(Vec<MersenneField>);

impl FieldVec {
    pub fn zeros(len: usize) -> FieldVec {
        FieldVec(vec![ZERO; len])
    }

    pub fn from_vec(v: Vec<MersenneField>) -> FieldVec {
        FieldVec(v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MersenneField> {
        self.0.iter()
    }

    pub fn as_slice(&self) -> &[MersenneField] {
        &self.0
    }
}

impl Index<usize> for FieldVec {
    type Output = MersenneField;
    fn index(&self, i: usize) -> &MersenneField {
        &self.0[i]
    }
}

impl IndexMut<usize> for FieldVec {
    fn index_mut(&mut self, i: usize) -> &mut MersenneField {
        &mut self.0[i]
    }
}

impl Add for FieldVec {
    type Output = FieldVec;
    fn add(self, other: FieldVec) -> FieldVec {
        assert_eq!(self.len(), other.len());
        FieldVec(
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }
}

impl Add<&FieldVec> for FieldVec {
    type Output = FieldVec;
    fn add(self, other: &FieldVec) -> FieldVec {
        assert_eq!(self.len(), other.len());
        FieldVec(
            self.0
                .into_iter()
                .zip(other.0.iter())
                .map(|(a, b)| a + *b)
                .collect(),
        )
    }
}

impl AddAssign<&FieldVec> for FieldVec {
    fn add_assign(&mut self, other: &FieldVec) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn f(v: u64) -> MersenneField {
        MersenneField::from_u64(v)
    }

    #[test]
    fn test_identity_add() {
        for v in [0u64, 1, 42, PRIME - 1, PRIME / 2] {
            let x = f(v);
            assert_eq!(ZERO + x, x, "ZERO + {v} != {v}");
        }
    }

    #[test]
    fn test_identity_mul() {
        for v in [0u64, 1, 42, PRIME - 1, PRIME / 2] {
            let x = f(v);
            assert_eq!(ONE * x, x, "ONE * {v} != {v}");
        }
    }

    #[test]
    fn test_commutativity_add() {
        let pairs = [(1u64, 2u64), (42, 100), (PRIME - 1, 1), (0, PRIME - 1)];
        for (a, b) in pairs {
            assert_eq!(f(a) + f(b), f(b) + f(a));
        }
    }

    #[test]
    fn test_commutativity_mul() {
        let pairs = [(3u64, 7u64), (42, 100), (PRIME - 1, 2), (0, 5)];
        for (a, b) in pairs {
            assert_eq!(f(a) * f(b), f(b) * f(a));
        }
    }

    #[test]
    fn test_associativity_add() {
        let triples = [(1u64, 2u64, 3u64), (42, 100, PRIME - 1), (0, 0, 0)];
        for (a, b, c) in triples {
            assert_eq!((f(a) + f(b)) + f(c), f(a) + (f(b) + f(c)));
        }
    }

    #[test]
    fn test_associativity_mul() {
        let triples = [(2u64, 3u64, 5u64), (42, 100, 7), (1, PRIME - 1, 2)];
        for (a, b, c) in triples {
            assert_eq!((f(a) * f(b)) * f(c), f(a) * (f(b) * f(c)));
        }
    }

    #[test]
    fn test_inverse_random() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let a: MersenneField = rng.sample(StandardUniform);
            if a == ZERO {
                continue;
            }
            assert_eq!(a * a.inverse().unwrap(), ONE, "inverse failed for {a}");
        }
    }

    #[test]
    fn test_inverse_zero() {
        assert_eq!(MersenneField(0).inverse(), None);
    }

    #[test]
    fn test_prime_minus_one_plus_one() {
        assert_eq!(MersenneField(PRIME - 1) + ONE, ZERO);
    }

    #[test]
    fn test_reduction_near_boundary() {
        // (PRIME-1) * 2 = 2*PRIME - 2 ≡ PRIME - 2 (mod PRIME)
        let result = MersenneField(PRIME - 1) * MersenneField(2);
        assert_eq!(result, MersenneField(PRIME - 2));
    }

    #[test]
    fn test_negation() {
        for v in [1u64, 42, 100, PRIME - 1, PRIME / 2] {
            let a = f(v);
            assert_eq!(-a + a, ZERO, "negation failed for {v}");
        }
        // Zero negation
        assert_eq!(-ZERO, ZERO);
    }

    #[test]
    fn test_subtraction() {
        let pairs = [(42u64, 7u64), (100, 100), (0, 0), (1, PRIME - 1)];
        for (a, b) in pairs {
            assert_eq!(
                f(a) - f(b) + f(b),
                f(a),
                "subtraction failed for ({a}, {b})"
            );
        }
    }

    #[test]
    fn test_from_i64_to_i64_roundtrip() {
        for v in [42i64, -7, 0] {
            let encoded = MersenneField::from_i64(v);
            assert_eq!(encoded.to_i64(), v, "round-trip failed for {v}");
        }
    }

    #[test]
    fn test_from_u64_reduces() {
        // Values >= PRIME should be reduced
        assert_eq!(MersenneField::from_u64(PRIME), ZERO);
        assert_eq!(MersenneField::from_u64(PRIME + 1), ONE);
        assert_eq!(MersenneField::from_u64(PRIME * 2), ZERO);
    }

    #[test]
    fn test_random_generation_range() {
        let mut rng = rand::rng();
        for _ in 0..1000 {
            let a: MersenneField = rng.sample(StandardUniform);
            assert!(a.inner() < PRIME, "sample {} >= PRIME", a.inner());
        }
    }

    #[test]
    fn test_fieldvec_zeros_length() {
        let v = FieldVec::zeros(3);
        assert_eq!(v.len(), 3);
        for i in 0..3 {
            assert_eq!(v[i], ZERO);
        }
    }

    #[test]
    fn test_fieldvec_addition() {
        let a = FieldVec::from_vec(vec![f(1), f(2), f(3)]);
        let b = FieldVec::from_vec(vec![f(10), f(20), f(30)]);
        let c = a + b;
        assert_eq!(c[0], f(11));
        assert_eq!(c[1], f(22));
        assert_eq!(c[2], f(33));
    }

    #[test]
    fn test_fieldvec_add_assign() {
        let mut a = FieldVec::from_vec(vec![f(1), f(2), f(3)]);
        let b = FieldVec::from_vec(vec![f(10), f(20), f(30)]);
        a += &b;
        assert_eq!(a[0], f(11));
        assert_eq!(a[1], f(22));
        assert_eq!(a[2], f(33));
    }

    #[test]
    #[should_panic]
    fn test_fieldvec_length_mismatch_panics() {
        let a = FieldVec::zeros(3);
        let b = FieldVec::zeros(4);
        let _ = a + b;
    }
}
