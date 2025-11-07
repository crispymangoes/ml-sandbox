use ndarray::{Array1, Array2};
use polars::prelude::*;

pub struct TitanicPreprocessor {
    pub df: DataFrame,
}

impl TitanicPreprocessor {
    /// Fit preprocessor on training data to learn normalization patterns.
    pub fn fit(df: DataFrame) -> Result<Self, PolarsError> {
        // Handle columns that may not be present
        let schema = df.schema();

        let survived_column_expr = if schema.contains("Survived") {
            col("Survived")
        } else {
            lit(LiteralValue::Null)
                .cast(DataType::Int32)
                .alias("Survived")
        };

        let lf = df.lazy();
        let processed_lf = lf
            // If age is missing, use median age
            .with_column(col("Age").fill_null(col("Age").median()))
            // Create new family size column
            .with_column((col("SibSp") + col("Parch") + lit(1)).alias("FamilySize"))
            // .filter(col("Age").is_not_null()) // I dont think this is needed
            .with_column(
                col("Cabin")
                    .str()
                    .slice(lit(0), lit(1))
                    .fill_null(lit("U"))
                    .alias("DeckLevel"),
            )
            .with_column(col("Embarked").fill_null(lit("S")))
            .with_column(col("FamilySize").eq(lit(1)).alias("IsAlone"))
            .with_column(
                col("Fare")
                    // Calculate the mean fare partitioned by Pclass
                    .fill_null(col("Fare").mean().over(["Pclass"]))
                    // Alias it back to Fare to overwrite original column
                    .alias("PreFare"),
            )
            // Make sure any zero fares are set to the mean fare for the class
            .with_column(
                when(col("PreFare").eq(lit(0.0)))
                    .then(col("PreFare").mean().over(["Pclass"]))
                    .otherwise(col("PreFare"))
                    // Alias it back to Fare to overwrite original column
                    .alias("Fare"),
            )
            .with_column(
                col("Name")
                    .str()
                    .extract(lit(r", (.*?)\."), 1) // think this means grab the first element
                    .alias("RawTitle"),
            )
            .with_column(
                when(col("RawTitle").eq(lit("Mr")))
                    .then(lit("Mr"))
                    // Match "Mrs" | "Mme" -> "Mrs" using .is_in()
                    .when(
                        col("RawTitle")
                            .is_in(lit(Series::new("titles_mrs".into(), &["Mrs", "Mme"]))),
                    )
                    .then(lit("Mrs"))
                    // Match "Miss" | "Mlle" | "Ms" -> "Miss" using .is_in()
                    .when(col("RawTitle").is_in(lit(Series::new(
                        "titles_miss".into(),
                        &["Miss", "Mlle", "Ms"],
                    ))))
                    .then(lit("Miss"))
                    // Handle master
                    .when(col("RawTitle").eq(lit("Master")))
                    .then(lit("Master"))
                    // Match "Dr" | "Rev" | "Col" | "Major" | "Capt" => "Officer"
                    .when(col("RawTitle").is_in(lit(Series::new(
                        "titles_officer".into(),
                        &["Dr", "Rev", "Col", "Major", "Capt"],
                    ))))
                    .then(lit("Officer"))
                    // Match "Lady" | "Countess" | "Jonkheer" | "Don" | "Sir" | "Dona" => "Royalty"
                    .when(col("RawTitle").is_in(lit(Series::new(
                        "titles_royalty".into(),
                        &["Lady", "Countess", "Jonkheer", "Don", "Sir", "Dona"],
                    ))))
                    .then(lit("Royalty"))
                    .otherwise(lit("Other"))
                    .alias("Title"),
            )
            .select([
                col("Pclass"),
                col("Sex"),
                col("Age"),
                col("Fare"),
                col("FamilySize"),
                col("DeckLevel"),
                col("Embarked"),
                col("IsAlone"),
                col("Title"),
                col("SibSp"),
                col("Parch"),
                survived_column_expr,
            ]);

        let result = processed_lf.collect()?;

        Ok(Self { df: result })
    }

    // Returns (features, Option<labels>)
    pub fn to_features(
        self,
        include_labels: bool,
    ) -> Result<(Array2<f32>, Option<Array1<f32>>), PolarsError> {
        // Features
        // Pclass -> 3, one hot encoded
        // Sex -> 1, one hot encoded
        // Age -> 1, scaled to min_age/max_age in list
        // Fare -> 1, scaled to min_fare/max_fare in list
        // FamilySize -> 1, scaled to min_family_size/max_family_size in list
        // DeckLevel -> 9(unique deck levels), one hot encoded
        // Embarked -> 3, one hot encoded
        // IsAlone -> 1, one hot encoded
        // Title -> 7(unique titles), one hot encoded
        // SibSp -> 1, scaled to min/max in list
        // Parch -> 1. scaled to min/max in list

        // TODO this grabs normalization params based off the data, but in reality this needs to be
        // repeated between train and test data, so I am thinking the fit function should accept the min/max params for each thing?
        let lf = self.df.lazy();
        let processed_lf = lf
            // Scale Age column
            .with_column((col("Age") - col("Age").mean()) / (col("Age").std(1)).alias("Age"))
            // Scale Fare column
            .with_column((col("Fare") - col("Fare").mean()) / (col("Fare").std(1)).alias("Fare"))
            // Convert Sex to binary
            .with_column(
                when(col("Sex").eq(lit("male")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Sex"),
            )
            // Convert Pclass to OHE
            .with_column(
                when(col("Pclass").eq(lit(1)))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Pclass1"),
            )
            .with_column(
                when(col("Pclass").eq(lit(2)))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Pclass2"),
            )
            .with_column(
                when(col("Pclass").eq(lit(3)))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Pclass3"),
            )
            // Convert FamilySize to f64
            .with_column(
                col("FamilySize")
                    .cast(DataType::Float64)
                    .alias("FamilySize"),
            )
            // Apply Z score to FamilySize
            .with_column(
                (col("FamilySize") - col("FamilySize").mean())
                    / (col("FamilySize").std(1)).alias("FamilySize"),
            )
            // Convert DeckLevel to OHE
            .with_column(
                when(col("DeckLevel").eq(lit("A")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel1"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("B")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel2"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("C")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel3"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("D")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel4"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("E")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel5"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("F")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel6"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("G")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel7"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("T")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel8"),
            )
            .with_column(
                when(col("DeckLevel").eq(lit("U")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("DeckLevel9"),
            )
            // Covnert Embarked to OHE
            .with_column(
                when(col("Embarked").eq(lit("Q")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Embarked1"),
            )
            .with_column(
                when(col("Embarked").eq(lit("C")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Embarked2"),
            )
            .with_column(
                when(col("Embarked").eq(lit("S")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Embarked3"),
            )
            // Convert IsAlone to bianry
            .with_column(
                when(col("IsAlone").eq(lit(true)))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("IsAlone"),
            )
            // Convert Title to OHE
            .with_column(
                when(col("Title").eq(lit("Other")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title1"),
            )
            .with_column(
                when(col("Title").eq(lit("Master")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title2"),
            )
            .with_column(
                when(col("Title").eq(lit("Miss")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title3"),
            )
            .with_column(
                when(col("Title").eq(lit("Mrs")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title4"),
            )
            .with_column(
                when(col("Title").eq(lit("Officer")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title5"),
            )
            .with_column(
                when(col("Title").eq(lit("Royalty")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title6"),
            )
            .with_column(
                when(col("Title").eq(lit("Mr")))
                    .then(lit(1.0))
                    .otherwise(lit(0.0))
                    .alias("Title7"),
            )
            // Convert SibSp to f64
            .with_column(col("SibSp").cast(DataType::Float64).alias("SibSp"))
            // Apply Z score to SibSp
            .with_column(
                (col("SibSp") - col("SibSp").mean()) / (col("SibSp").std(1)).alias("SibSp"),
            )
            // Convert Parch to f64
            .with_column(col("Parch").cast(DataType::Float64).alias("Parch"))
            // Apply Z score to Parch
            .with_column(
                (col("Parch") - col("Parch").mean()) / (col("Parch").std(1)).alias("Parch"),
            )
            .select([
                col("Pclass1"),
                col("Pclass2"),
                col("Pclass3"),
                col("Sex"),
                col("Age"),
                col("Fare"),
                col("FamilySize"),
                col("DeckLevel1"),
                col("DeckLevel2"),
                col("DeckLevel3"),
                col("DeckLevel4"),
                col("DeckLevel5"),
                col("DeckLevel6"),
                col("DeckLevel7"),
                col("DeckLevel8"),
                col("DeckLevel9"),
                col("Embarked1"),
                col("Embarked2"),
                col("Embarked3"),
                col("IsAlone"),
                col("Title1"),
                col("Title2"),
                col("Title3"),
                col("Title4"),
                col("Title5"),
                col("Title6"),
                col("Title7"),
                col("SibSp"),
                col("Parch"),
                col("Survived"),
            ]);

        let mut result = processed_lf.collect()?;

        println!("{}", result.head(Some(10)));

        let mut file = std::fs::File::create("output.csv")?;
        CsvWriter::new(&mut file)
            .include_header(true) // Include header row
            .with_separator(b',') // Set separator to comma (default)
            .with_null_value("NULL".to_string()) // Represent nulls as "NULL"
            .finish(&mut result)?; // Write the DataFrame

        let n_rows = result.height();
        let n_cols = result.width() - 1; // Exclude survived

        println!("Converting to ndarray...");
        println!("  Rows: {}", n_rows);
        println!("  Feature columns: {}", n_cols);

        let mut features = Array2::<f32>::zeros((n_rows, n_cols));

        // Fill feature matrix column by column
        for col_idx in 0..n_cols {
            let col = result.get_columns()[col_idx].cast(&DataType::Float64)?;
            let values = col.f64()?;

            for row_idx in 0..n_rows {
                features[[row_idx, col_idx]] = values.get(row_idx).unwrap_or(0.0) as f32;
            }
        }

        // Extract labels if requested
        let labels = if include_labels {
            let survived_col = result.column("Survived")?.cast(&DataType::Float64)?;
            let survived_values = survived_col.f64()?;

            let mut labels_array = Array1::<f32>::zeros(n_rows);
            for i in 0..n_rows {
                labels_array[i] = survived_values.get(i).unwrap_or(0.0) as f32;
            }

            Some(labels_array)
        } else {
            None
        };

        println!("Conversion complete!");
        println!(
            "  Feature matrix shape: {} x {}",
            features.nrows(),
            features.ncols()
        );
        if let Some(ref l) = labels {
            println!("  Labels shape: {}", l.len());
            let survival_rate = l.iter().sum::<f32>() / l.len() as f32;
            println!("  Survival rate: {:.2}%", survival_rate * 100.0);
        }

        Ok((features, labels))
    }

    pub fn unique_decks(&self) -> Result<usize, PolarsError> {
        let unique_decks = self.df.column("DeckLevel")?.unique()?;
        Ok(unique_decks.len())
    }

    pub fn unique_titles(&self) -> Result<usize, PolarsError> {
        let unique_titles = self.df.column("Title")?.unique()?;
        Ok(unique_titles.len())
    }

    pub fn get_unique_titles(&self) -> Result<Column, PolarsError> {
        Ok(self.df.column("Title")?.unique()?)
    }

    pub fn get_unique_decks(&self) -> Result<Column, PolarsError> {
        Ok(self.df.column("DeckLevel")?.unique()?)
    }

    /// Get the total number of features this preprocessor will produce
    pub fn feature_count(&self) -> Result<usize, PolarsError> {
        let unique_decks = self.unique_decks()?;
        let unique_titles = self.unique_titles()?;

        // Pclass: 3, Sex: 1, Age: 1, Fare: 1, FamilySize: 1,
        // DeckLevel: unique_decks, Embarked: 3, IsAlone: 1,
        // Title: unique_titles, SibSp: 1, Parch: 1
        let count = 3 + 1 + 1 + 1 + 1 + unique_decks + 3 + 1 + unique_titles + 1 + 1;

        Ok(count)
    }

    /// Print a summary of the feature engineering
    pub fn print_feature_summary(&self) -> Result<(), PolarsError> {
        let unique_decks = self.unique_decks()?;
        let unique_titles = self.unique_titles()?;
        let total = self.feature_count()?;

        println!("\n=== Feature Engineering Summary ===");
        println!("Pclass (one-hot):        3 features");
        println!("Sex (binary):            1 feature");
        println!("Age (normalized):        1 feature");
        println!("Fare (normalized):       1 feature");
        println!("FamilySize (normalized): 1 feature");
        println!("DeckLevel (one-hot):     {} features", unique_decks);
        println!("Embarked (one-hot):      3 features");
        println!("IsAlone (binary):        1 feature");
        println!("Title (one-hot):         {} features", unique_titles);
        println!("SibSp (normalized):      1 feature");
        println!("Parch (normalized):      1 feature");
        println!("-----------------------------------");
        println!("TOTAL:                   {} features\n", total);

        Ok(())
    }
}
