use crate::data::{TitanicBatch, TitanicBatcher};
use crate::dataset::TitanicDataset;
use crate::model::{TitanicModel, TitanicModelConfig};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    nn::loss::BinaryCrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl<B: Backend> TitanicModel<B> {
    /// Forward pass with loss calculation for training
    pub fn forward_classification(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 1, burn::prelude::Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(input);

        // BCE loss expects Int targets in 2D shape
        let targets_2d = targets.clone().reshape([targets.dims()[0], 1]);

        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets_2d);

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TitanicBatch<B>, ClassificationOutput<B>> for TitanicModel<B> {
    fn step(&self, batch: TitanicBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TitanicBatch<B>, ClassificationOutput<B>> for TitanicModel<B> {
    fn step(&self, batch: TitanicBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TitanicTrainingConfig {
    pub model: TitanicModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1000)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-5)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TitanicTrainingConfig,
    dataset: TitanicDataset,
    device: B::Device,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TitanicBatcher::default();

    // Split dataset
    let (train_dataset, val_dataset) = dataset.split(0.8);

    let dataloader_train: std::sync::Arc<dyn DataLoader<B, TitanicBatch<B>>> =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(val_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
