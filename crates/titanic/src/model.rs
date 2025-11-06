use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{Tensor, activation, backend::Backend},
};

#[derive(Module, Debug)]
pub struct TitanicModel<B: Backend> {
    fc1: Linear<B>,
    dropout1: Dropout,
    fc2: Linear<B>,
    dropout2: Dropout,
    fc3: Linear<B>,
    dropout3: Dropout,
    output: Linear<B>,
}

#[derive(Config, Debug)]
pub struct TitanicModelConfig {
    pub input_size: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub hidden3: usize,
    pub dropout_rate: f64,
}

impl Default for TitanicModelConfig {
    fn default() -> Self {
        Self {
            input_size: 29,    // Based on your feature engineering
            hidden1: 64,       // First hidden layer
            hidden2: 32,       // Second hidden layer
            hidden3: 16,       // Third hidden layer
            dropout_rate: 0.3, // 30% dropout to prevent overfitting
        }
    }
}

impl TitanicModelConfig {
    // pub fn new(
    //     input_size: usize,
    //     hidden1: usize,
    //     hidden2: usize,
    //     hidden3: usize,
    //     dropout_rate: f64,
    // ) -> Self {
    //     Self {
    //         input_size,
    //         hidden1,
    //         hidden2,
    //         hidden3,
    //         dropout_rate,
    //     }
    // }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TitanicModel<B> {
        TitanicModel {
            fc1: LinearConfig::new(self.input_size, self.hidden1).init(device),
            dropout1: DropoutConfig::new(self.dropout_rate).init(),
            fc2: LinearConfig::new(self.hidden1, self.hidden2).init(device),
            dropout2: DropoutConfig::new(self.dropout_rate).init(),
            fc3: LinearConfig::new(self.hidden2, self.hidden3).init(device),
            dropout3: DropoutConfig::new(self.dropout_rate).init(),
            output: LinearConfig::new(self.hidden3, 1).init(device),
        }
    }
}

impl<B: Backend> TitanicModel<B> {
    /// Forward pass through the network
    /// Returns probabilities in range [0, 1]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // First layer: input -> hidden1
        let x = self.fc1.forward(input);
        let x = activation::relu(x);
        let x = self.dropout1.forward(x);

        // Second layer: hidden1 -> hidden2
        let x = self.fc2.forward(x);
        let x = activation::relu(x);
        let x = self.dropout2.forward(x);

        // Third layer: hidden2 -> hidden3
        let x = self.fc3.forward(x);
        let x = activation::relu(x);
        let x = self.dropout3.forward(x);

        // Output layer: hidden3 -> 1 (probability)
        let x = self.output.forward(x);
        activation::sigmoid(x) // Sigmoid to get probability [0, 1]
    }

    /// Make predictions (0 or 1) from input features
    pub fn predict(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let probs = self.forward(input);
        // Convert probabilities to binary predictions (threshold at 0.5)
        probs.greater_elem(0.5).float()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use burn::backend::ndarray::NdArray;
//     use burn::tensor::{Tensor, TensorData};

//     type TestBackend = NdArray<f32>;

//     #[test]
//     fn test_model_creation() {
//         let device = Default::default();
//         let config = TitanicModelConfig::default();
//         let model = config.init::<TestBackend>(&device);

//         // Create dummy input (batch_size=2, features=29)
//         let input = Tensor::<TestBackend, 2>::from_floats([[0.5; 29], [0.3; 29]], &device);

//         let output = model.forward(input);
//         let shape = output.shape();

//         assert_eq!(shape.dims[0], 2); // batch size
//         assert_eq!(shape.dims[1], 1); // output size
//     }

//     #[test]
//     fn test_output_range() {
//         let device = Default::default();
//         let config = TitanicModelConfig::default();
//         let model = config.init::<TestBackend>(&device);

//         let input = Tensor::<TestBackend, 2>::from_floats([[0.5; 29]], &device);

//         let output = model.forward(input);
//         let output_data: TensorData = output.into_data();
//         let values: Vec<f32> = output_data.into();
//         let value = values[0];

//         // Sigmoid output should be between 0 and 1
//         assert!(value >= 0.0 && value <= 1.0);
//     }
// }
