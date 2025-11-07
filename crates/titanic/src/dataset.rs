use crate::data::TitanicItem;
use burn::data::dataset::Dataset;

/// Dataset for Titanic training data
#[derive(Clone)]
pub struct TitanicDataset {
    features: Vec<Vec<f32>>,
    labels: Vec<i32>,
}

impl TitanicDataset {
    pub fn new(features: ndarray::Array2<f32>, labels: ndarray::Array1<f32>) -> Self {
        let n_rows = features.nrows();
        let n_cols = features.ncols();

        // Covnert to Vec<Vec<f32>>
        let mut features_vec = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f32> = (0..n_cols).map(|j| features[[i, j]]).collect();
            features_vec.push(row);
        }

        // Convert labels to i32
        let labels_vec: Vec<i32> = labels.iter().map(|&x| x as i32).collect();

        Self {
            features: features_vec,
            labels: labels_vec,
        }
    }

    pub fn new_no_labels(features: ndarray::Array2<f32>) -> Self {
        let n_rows = features.nrows();
        let n_cols = features.ncols();

        // Covnert to Vec<Vec<f32>>
        let mut features_vec = Vec::with_capacity(n_rows);
        let mut empty_vec = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f32> = (0..n_cols).map(|j| features[[i, j]]).collect();
            features_vec.push(row);
            empty_vec.push(0);
        }

        Self {
            features: features_vec,
            labels: empty_vec,
        }
    }

    /// Create train/val split
    pub fn split(self, train_ratio: f32) -> (Self, Self) {
        let n_total = self.features.len();
        let n_train = (n_total as f32 * train_ratio) as usize;

        let train_features = self.features[..n_train].to_vec();
        let train_labels = self.labels[..n_train].to_vec();

        let val_features = self.features[n_train..].to_vec();
        let val_labels = self.labels[n_train..].to_vec();

        let train_dataset = Self {
            features: train_features,
            labels: train_labels,
        };

        let val_dataset = Self {
            features: val_features,
            labels: val_labels,
        };

        (train_dataset, val_dataset)
    }
}

impl Dataset<TitanicItem> for TitanicDataset {
    fn get(&self, index: usize) -> Option<TitanicItem> {
        if index > self.features.len() {
            return None;
        }

        Some(TitanicItem::new(
            self.features[index].clone(),
            self.labels[index],
        ))
    }

    fn len(&self) -> usize {
        self.features.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_dataset_creation() {
        let features = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let dataset = TitanicDataset::new(features, labels);

        assert_eq!(dataset.len(), 5);

        let item = dataset.get(0).unwrap();
        assert_eq!(item.features, vec![1.0, 2.0, 3.0]);
        assert_eq!(item.label, 0);
    }

    #[test]
    fn test_dataset_split() {
        let features = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();

        let labels = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let dataset = TitanicDataset::new(features, labels);
        let (train, val) = dataset.split(0.8);

        assert_eq!(train.len(), 4);
        assert_eq!(val.len(), 1);
    }
}
