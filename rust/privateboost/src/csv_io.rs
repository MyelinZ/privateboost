use crate::Result;
use std::path::Path;

pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    pub feature_names: Vec<String>,
}

pub fn read_csv(path: &Path, target_column: &str) -> Result<Dataset> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader.headers()?.iter().map(String::from).collect();

    let target_idx = headers
        .iter()
        .position(|h| h == target_column)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("target column '{}' not found", target_column),
            )
        })?;

    let feature_names: Vec<String> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target_idx)
        .map(|(_, h)| h.clone())
        .collect();

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row_features = Vec::with_capacity(feature_names.len());
        let mut target = 0.0;

        for (i, field) in record.iter().enumerate() {
            let value: f64 = field.parse().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("cannot parse '{}' as f64", field),
                )
            })?;
            if i == target_idx {
                target = value;
            } else {
                row_features.push(value);
            }
        }

        features.push(row_features);
        targets.push(target);
    }

    Ok(Dataset {
        features,
        targets,
        feature_names,
    })
}

pub fn write_results(path: &Path, predictions: &[f64], targets: &[f64]) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["prediction", "target"])?;
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        writer.write_record(&[pred.to_string(), target.to_string()])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_csv() {
        let dir = std::env::temp_dir().join("privateboost_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "a,b,target").unwrap();
        writeln!(f, "1.0,2.0,0").unwrap();
        writeln!(f, "3.0,4.0,1").unwrap();

        let ds = read_csv(&path, "target").unwrap();
        assert_eq!(ds.feature_names, vec!["a", "b"]);
        assert_eq!(ds.features.len(), 2);
        assert_eq!(ds.targets, vec![0.0, 1.0]);
        assert_eq!(ds.features[0], vec![1.0, 2.0]);
    }
}
