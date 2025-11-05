pub mod titanic_preprocessor;
use crate::titanic_preprocessor::TitanicPreprocessor;
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

    preprocessor.to_features(false)?;

    Ok(())
}
