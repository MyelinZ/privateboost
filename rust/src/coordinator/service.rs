use std::collections::HashMap;
use std::sync::Mutex;
use uuid::Uuid;

use crate::domain::model::RunConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Active,
    Cancelled,
    Complete,
}

pub struct Coordinator {
    runs: Mutex<HashMap<String, (RunConfig, RunStatus)>>,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_run(&self, mut config: RunConfig) -> String {
        let run_id = Uuid::new_v4().to_string();
        config.run_id = run_id.clone();
        let mut runs = self.runs.lock().unwrap();
        runs.insert(run_id.clone(), (config, RunStatus::Active));
        run_id
    }

    pub fn cancel_run(&self, run_id: &str) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(entry) = runs.get_mut(run_id) {
            entry.1 = RunStatus::Cancelled;
        }
    }

    pub fn complete_run(&self, run_id: &str) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(entry) = runs.get_mut(run_id) {
            entry.1 = RunStatus::Complete;
        }
    }

    pub fn list_runs(&self) -> Vec<(String, RunStatus)> {
        let runs = self.runs.lock().unwrap();
        runs.iter()
            .map(|(id, (_, status))| (id.clone(), *status))
            .collect()
    }

    pub fn get_run_config(&self, run_id: &str) -> Option<RunConfig> {
        let runs = self.runs.lock().unwrap();
        runs.get(run_id).map(|(config, _)| config.clone())
    }

    pub fn get_run_status(&self, run_id: &str) -> Option<RunStatus> {
        let runs = self.runs.lock().unwrap();
        runs.get(run_id).map(|(_, status)| *status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> RunConfig {
        RunConfig {
            run_id: String::new(),
            n_bins: 10,
            threshold: 2,
            min_clients: 5,
            learning_rate: 0.15,
            lambda_reg: 2.0,
            n_trees: 3,
            max_depth: 3,
            loss: "squared".into(),
            target_count: 5,
            features: vec!["f1".into(), "f2".into()],
            target_column: "target".into(),
        }
    }

    #[test]
    fn test_create_run() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config.clone());
        assert!(!run_id.is_empty());
        let fetched = coord.get_run_config(&run_id).unwrap();
        assert_eq!(fetched.n_bins, config.n_bins);
        assert_eq!(fetched.run_id, run_id);
    }

    #[test]
    fn test_list_runs() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config);
        let runs = coord.list_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].0, run_id);
        assert_eq!(runs[0].1, RunStatus::Active);
    }

    #[test]
    fn test_cancel_run() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config);
        coord.cancel_run(&run_id);
        let status = coord.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Cancelled);
    }

    #[test]
    fn test_complete_run() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config);
        coord.complete_run(&run_id);
        let status = coord.get_run_status(&run_id).unwrap();
        assert_eq!(status, RunStatus::Complete);
    }

    #[test]
    fn test_get_unknown_run() {
        let coord = Coordinator::new();
        assert!(coord.get_run_config("unknown").is_none());
        assert!(coord.get_run_status("unknown").is_none());
    }
}
