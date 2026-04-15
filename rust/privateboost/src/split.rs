use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;

pub struct SplitData {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

pub fn stratified_split(targets: &[f64], test_fraction: f64, seed: u64) -> SplitData {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut pos_indices: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 1.0)
        .map(|(i, _)| i)
        .collect();
    let mut neg_indices: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 0.0)
        .map(|(i, _)| i)
        .collect();

    pos_indices.shuffle(&mut rng);
    neg_indices.shuffle(&mut rng);

    let n_pos_test = (pos_indices.len() as f64 * test_fraction).round() as usize;
    let n_neg_test = (neg_indices.len() as f64 * test_fraction).round() as usize;

    let mut test_indices = Vec::new();
    let mut train_indices = Vec::new();

    test_indices.extend_from_slice(&pos_indices[..n_pos_test]);
    train_indices.extend_from_slice(&pos_indices[n_pos_test..]);
    test_indices.extend_from_slice(&neg_indices[..n_neg_test]);
    train_indices.extend_from_slice(&neg_indices[n_neg_test..]);

    SplitData {
        train_indices,
        test_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_split_preserves_ratio() {
        let targets = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let split = stratified_split(&targets, 0.2, 42);

        let train_pos = split.train_indices.iter().filter(|&&i| targets[i] == 1.0).count();
        let test_pos = split.test_indices.iter().filter(|&&i| targets[i] == 1.0).count();

        assert_eq!(test_pos, 1);
        assert_eq!(train_pos, 3);
        assert_eq!(split.train_indices.len() + split.test_indices.len(), 10);
    }

    #[test]
    fn test_stratified_split_deterministic() {
        let targets = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let s1 = stratified_split(&targets, 0.33, 7);
        let s2 = stratified_split(&targets, 0.33, 7);
        assert_eq!(s1.train_indices, s2.train_indices);
        assert_eq!(s1.test_indices, s2.test_indices);
    }
}
