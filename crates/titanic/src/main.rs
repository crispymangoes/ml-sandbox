mod data;
mod dataset;
mod inference;
mod model;
mod titanic_preprocessor;
mod training;
use crate::inference::infer;
use crate::titanic_preprocessor::TitanicPreprocessor;
use crate::{dataset::TitanicDataset, model::TitanicModelConfig, training::TitanicTrainingConfig};
use burn::data::dataset::Dataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    optim::AdamConfig,
};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Titanic Training Pipeline ===\n");

    // Load training data
    println!("Loading training data...");
    let file = std::fs::File::open("./data/raw/train.csv")?;
    let df = CsvReader::new(file).finish()?;
    println!("Loaded {} rows\n", df.height());

    // Fit preprocessor
    println!("Fitting preprocessor...");
    let preprocessor = TitanicPreprocessor::fit(df)?;

    println!("Processed dataframe");
    println!("{}", preprocessor.df.head(Some(10)));

    // Print feature summary
    preprocessor.print_feature_summary()?;

    let unique_decks = preprocessor.get_unique_titles()?;
    println!("{:?}", unique_decks);

    let (features, labels) = preprocessor.to_features(true)?;
    let labels = labels.unwrap();

    let dataset = TitanicDataset::new(features, labels);

    // type MyBackend = Wgpu<f32, i32>;
    type MyBackend = NdArray<f64, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let artifact_dir = "./tmp/guide";
    let gradient_clipping_config = GradientClippingConfig::Value(0.1);
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TitanicTrainingConfig::new(
            TitanicModelConfig::default(),
            AdamConfig::new().with_grad_clipping(Some(gradient_clipping_config)),
        ),
        dataset.clone(),
        device.clone(),
    );

    // infer::<MyBackend>(artifact_dir, device, dataset.get(2).unwrap());

    // Load testing data
    println!("Loading testing data...");
    let file = std::fs::File::open("./data/raw/test.csv")?;
    let df = CsvReader::new(file).finish()?;
    println!("Loaded {} rows\n", df.height());

    // Fit preprocessor
    println!("Fitting preprocessor...");
    let preprocessor = TitanicPreprocessor::fit(df)?;

    let (features, _labels) = preprocessor.to_features(false)?;

    let test_dataset = TitanicDataset::new_no_labels(features);

    let mut passenger_id = 892;
    let mut predictions = Vec::new();
    let mut right = 0;
    let mut wrong = 0;
    for i in 0..test_dataset.len() {
        let (prediction, label) =
            infer::<MyBackend>(artifact_dir, device, test_dataset.get(i).unwrap());
        predictions.push((passenger_id, prediction));
        passenger_id += 1;
        if prediction == label {
            right += 1;
        } else {
            wrong += 1;
        }
    }

    let score = right as f32 / test_dataset.len() as f32;
    println!("Model Accuracy on Test Data: {}", score);

    let output_path = "submission.csv";
    let mut wtr = csv::Writer::from_path(output_path)?;

    // Write the header and all records
    for record in predictions {
        wtr.serialize(record)?;
    }
    wtr.flush()?;

    Ok(())
}
