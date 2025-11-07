use crate::data::{TitanicBatch, TitanicBatcher};
use crate::dataset::TitanicDataset;
use crate::model::{TitanicModel, TitanicModelConfig};
use burn::tensor::activation;
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
    /// The issue was the accuracy metric is designed to handle an arbitrary number of output classes
    /// so it looks at each output column and takes the one with the highest value
    /// but since my model was only outputting a single column, it always picked the single column as the highest
    /// meaning that is always chose 0 did not survive for its answer
    /// so the fix was to make the single column output into 2 columns and give that to the Classification output!
    pub fn forward_classification(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 1, burn::prelude::Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(input); // Logits share: [N, 1]

        // Covnert targets to float and reshape
        let targets_2d = targets.clone().reshape([targets.dims()[0], 1]);

        // Use sigmoid + BCE
        // let probs = activation::sigmoid(logits.clone());
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits.clone(), targets_2d);

        let probs_class_1 = activation::sigmoid(logits.clone());
        let probs_class_0 = 1.0 - probs_class_1.clone();

        // Concatenate them to create a [N, 2] tensor: [[P(0), P(1)], ...]
        let metric_output = Tensor::cat(vec![probs_class_0, probs_class_1], 1);

        // Pass the new [N, 2] output to ClassificationOutput
        ClassificationOutput::new(loss, metric_output, targets)
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
    #[config(default = 27)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
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
    println!("Learning Rate: {}", config.learning_rate);
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
