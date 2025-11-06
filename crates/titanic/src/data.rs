use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{Tensor, TensorData, backend::Backend},
};

#[derive(Debug, Clone)]
pub struct TitanicItem {
    pub features: Vec<f32>, // 29 features
    pub label: i32,         // 0 or 1
}

impl TitanicItem {
    pub fn new(features: Vec<f32>, label: i32) -> Self {
        Self { features, label }
    }
}

/// Batch structure for training
#[derive(Clone, Debug)]
pub struct TitanicBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, burn::prelude::Int>,
}

impl<B: Backend> TitanicBatch<B> {
    /// Create a new batch from features and labels
    pub fn new(inputs: Tensor<B, 2>, targets: Tensor<B, 1, burn::prelude::Int>) -> Self {
        Self { inputs, targets }
    }
}

/// Batcher converts individual titanic items into batches
#[derive(Clone, Default)]
pub struct TitanicBatcher {}

impl TitanicBatcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl<B: Backend> Batcher<B, TitanicItem, TitanicBatch<B>> for TitanicBatcher {
    fn batch(&self, items: Vec<TitanicItem>, device: &B::Device) -> TitanicBatch<B> {
        let batch_size = items.len();
        let num_features = items[0].features.len();

        // Collect all features into a flat vector
        let mut features_flat = Vec::with_capacity(batch_size * num_features);
        let mut labels = Vec::with_capacity(batch_size);

        for item in items.iter() {
            features_flat.extend_from_slice(&item.features);
            labels.push(item.label);
        }

        // Create tensors
        let inputs = Tensor::<B, 2>::from_floats(
            TensorData::new(features_flat, [batch_size, num_features]),
            device,
        );

        let targets = Tensor::<B, 1, burn::prelude::Int>::from_ints(
            TensorData::new(labels, [batch_size]),
            device,
        );

        TitanicBatch::new(inputs, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::ndarray::NdArrayDevice::Cpu;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_batcher() {
        let batcher = TitanicBatcher::new();

        // Create some test items
        let items = vec![
            TitanicItem::new(vec![0.5; 29], 1),
            TitanicItem::new(vec![0.3; 29], 0),
            TitanicItem::new(vec![0.7; 29], 1),
        ];

        let batch: TitanicBatch<TestBackend> = batcher.batch(items, &Cpu);

        assert_eq!(batch.inputs.dims(), [3, 29]);
        assert_eq!(batch.targets.dims(), [3]);
    }
}
