pub mod data;
pub mod dataset;
pub mod model;
pub mod titanic_preprocessor;
pub mod training;
use crate::titanic_preprocessor::TitanicPreprocessor;
use crate::{dataset::TitanicDataset, model::TitanicModelConfig, training::TitanicTrainingConfig};
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
        dataset,
        device.clone(),
    );

    Ok(())
}
