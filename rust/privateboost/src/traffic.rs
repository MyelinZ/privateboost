use serde::Serialize;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    ClientToShareholder,
    ShareholderToAggregator,
    AggregatorBroadcast,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::ClientToShareholder => write!(f, "ClientToShareholder"),
            Direction::ShareholderToAggregator => write!(f, "ShareholderToAggregator"),
            Direction::AggregatorBroadcast => write!(f, "AggregatorBroadcast"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrafficEntry {
    pub tree: usize,
    pub depth: usize,
    pub direction: Direction,
    pub message_type: &'static str,
    pub bytes: u64,
    pub count: u64,
}

#[derive(Clone, Debug, Default)]
pub struct TrafficLog {
    entries: Vec<TrafficEntry>,
}

impl TrafficLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a batch of identical messages.
    /// Serializes `msg` once and multiplies by `count`.
    pub fn record<T: Serialize>(
        &mut self,
        msg: &T,
        direction: Direction,
        tree: usize,
        depth: usize,
        msg_type: &'static str,
        count: u64,
    ) {
        let size = bincode::serialized_size(msg).unwrap();
        self.entries.push(TrafficEntry {
            tree,
            depth,
            direction,
            message_type: msg_type,
            bytes: size * count,
            count,
        });
    }

    pub fn entries(&self) -> &[TrafficEntry] {
        &self.entries
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    pub fn total_c2s(&self) -> u64 {
        self.entries.iter()
            .filter(|e| e.direction == Direction::ClientToShareholder)
            .map(|e| e.bytes).sum()
    }

    pub fn total_s2a(&self) -> u64 {
        self.entries.iter()
            .filter(|e| e.direction == Direction::ShareholderToAggregator)
            .map(|e| e.bytes).sum()
    }

    pub fn total_broadcast(&self) -> u64 {
        self.entries.iter()
            .filter(|e| e.direction == Direction::AggregatorBroadcast)
            .map(|e| e.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::Commitment;

    #[derive(serde::Serialize)]
    struct FakeMessage {
        x: u64,
        data: Vec<u64>,
    }

    #[test]
    fn test_record_and_totals() {
        let mut log = TrafficLog::new();
        let msg = FakeMessage { x: 1, data: vec![10, 20, 30] };
        let size = bincode::serialized_size(&msg).unwrap();

        log.record(&msg, Direction::ClientToShareholder, 0, 0, "FakeMessage", 10);

        assert_eq!(log.total_c2s(), size * 10);
        assert_eq!(log.total_s2a(), 0);
        assert_eq!(log.total_broadcast(), 0);
        assert_eq!(log.total_bytes(), size * 10);
    }

    #[test]
    fn test_multiple_directions() {
        let mut log = TrafficLog::new();
        let msg = FakeMessage { x: 1, data: vec![10] };
        let size = bincode::serialized_size(&msg).unwrap();

        log.record(&msg, Direction::ClientToShareholder, 0, 0, "FakeMessage", 5);
        log.record(&msg, Direction::ShareholderToAggregator, 0, 0, "FakeMessage", 3);
        log.record(&msg, Direction::AggregatorBroadcast, 0, 0, "FakeMessage", 2);

        assert_eq!(log.total_c2s(), size * 5);
        assert_eq!(log.total_s2a(), size * 3);
        assert_eq!(log.total_broadcast(), size * 2);
        assert_eq!(log.total_bytes(), size * 10);
    }

    #[test]
    fn test_entries_per_round() {
        let mut log = TrafficLog::new();
        let msg = FakeMessage { x: 1, data: vec![10] };

        log.record(&msg, Direction::ClientToShareholder, 0, 0, "FakeMessage", 1);
        log.record(&msg, Direction::ClientToShareholder, 0, 1, "FakeMessage", 1);
        log.record(&msg, Direction::ClientToShareholder, 1, 0, "FakeMessage", 1);

        assert_eq!(log.entries().len(), 3);
    }

    #[test]
    fn test_serialized_size_matches_bincode() {
        let commitment = Commitment([0xAB; 32]);
        let expected = bincode::serialized_size(&commitment).unwrap();
        assert_eq!(expected, 32);
    }
}
